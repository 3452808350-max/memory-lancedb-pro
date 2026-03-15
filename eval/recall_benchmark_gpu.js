#!/usr/bin/env node
/**
 * Synthetic PerLTQA 召回率基准测试 - 本地 GPU 加速版
 * 使用 AMD RX 6800 ROCm 加速 embedding 计算
 */

import * as lancedb from '@lancedb/lancedb';
import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';

// ============ 配置 ============
const config = {
  dbPath: '/home/kyj/.openclaw/workspace/lancedb',
  testScope: `perltqa-gpu-test-${Date.now()}`,
  dataDir: '/home/kyj/.openclaw/workspace/synthetic_perltqa'
};

// 数据规模梯度
const SCALES = ['baseline', 'small', 'medium', 'medium-large', 'large'];

// ============ Python Embedding 服务 ============

class GPUEmbeddingService {
  constructor() {
    this.python = null;
    this.ready = false;
    this.requestId = 0;
    this.pending = new Map();
  }

  async start() {
    return new Promise((resolve, reject) => {
      // 启动 Python GPU embedding 服务
      this.python = spawn('python3', ['-u', '-c', `
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-m3', device='cuda')
batch_size = 128

while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        data = json.loads(line.strip())
        texts = data['texts']
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            emb = model.encode(batch, batch_size=batch_size, show_progress_bar=False)
            embeddings.extend(emb.tolist())
        print(json.dumps({'id': data['id'], 'embeddings': embeddings}), flush=True)
    except Exception as e:
        print(json.dumps({'id': data['id'], 'error': str(e)}), flush=True)
      `], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.python.stdout.on('data', (data) => {
        const lines = data.toString().split('\n').filter(l => l.trim());
        for (const line of lines) {
          try {
            const result = JSON.parse(line);
            const pending = this.pending.get(result.id);
            if (pending) {
              this.pending.delete(result.id);
              if (result.error) {
                pending.reject(new Error(result.error));
              } else {
                pending.resolve(result.embeddings);
              }
            }
          } catch (e) {
            console.error('[Embedding] Parse error:', e);
          }
        }
      });

      this.python.stderr.on('data', (data) => {
        console.error('[Embedding]', data.toString());
      });

      this.python.on('error', reject);
      this.python.on('exit', (code) => {
        console.log(`[Embedding] Python exited with code ${code}`);
        this.ready = false;
      });

      // 测试连接
      setTimeout(() => {
        this.ready = true;
        resolve();
      }, 3000);
    });
  }

  async getEmbeddings(texts) {
    if (!this.ready) {
      throw new Error('Embedding service not ready');
    }

    return new Promise((resolve, reject) => {
      const id = ++this.requestId;
      this.pending.set(id, { resolve, reject });
      this.python.stdin.write(JSON.stringify({ id, texts }) + '\n');
      
      // 超时保护
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error('Embedding timeout'));
        }
      }, 60000);
    });
  }

  async stop() {
    if (this.python) {
      this.python.stdin.end();
      this.python.kill();
    }
  }
}

// ============ 核心测试逻辑 ============

class RecallMetrics {
  constructor() {
    this.results = [];
  }

  record(query, targetId, retrieved) {
    const hit = retrieved.slice(0, 5).some(r => String(r.id) === String(targetId));
    const rank = retrieved.findIndex(r => String(r.id) === String(targetId)) + 1;
    
    this.results.push({
      query_id: query.id,
      query: query.query,
      target_id: targetId,
      hit_at_5: hit,
      rank: rank > 0 ? rank : -1
    });
  }

  getRecallAtK(k = 5) {
    const hits = this.results.filter(r => r.hit_at_5).length;
    return hits / this.results.length;
  }

  getMRR() {
    let sum = 0;
    for (const r of this.results) {
      if (r.rank > 0) {
        sum += 1 / r.rank;
      }
    }
    return sum / this.results.length;
  }

  getSummary() {
    return {
      total_queries: this.results.length,
      recall_at_5: this.getRecallAtK(5).toFixed(3),
      recall_at_10: this.getRecallAtK(10).toFixed(3),
      mrr: this.getMRR().toFixed(3)
    };
  }
}

async function insertMemories(db, memories, scope, embeddingService) {
  console.log(`  📥 插入 ${memories.length} 条记忆到 scope: ${scope}...`);
  
  // 先删除旧表
  try {
    await db.dropTable(scope);
  } catch (e) {}
  
  // 批量生成 embedding
  console.log(`  🚀 生成 embedding (GPU 加速)...`);
  const startTime = Date.now();
  
  const batchSize = 500;
  const allRecords = [];
  
  for (let i = 0; i < memories.length; i += batchSize) {
    const batch = memories.slice(i, i + batchSize);
    const texts = batch.map(m => m.content);
    const embeddings = await embeddingService.getEmbeddings(texts);
    
    const records = batch.map((m, idx) => ({
      id: m.id,
      text: m.content,
      embedding: embeddings[idx],
      category: m.type,
      importance: 0.7,
      timestamp: m.timestamp
    }));
    
    allRecords.push(...records);
    
    const progress = Math.min(i + batchSize, memories.length);
    process.stdout.write(`\r  📊 进度：${progress}/${memories.length} (${((progress/memories.length)*100).toFixed(0)}%)`);
  }
  
  const embedTime = ((Date.now() - startTime) / 1000).toFixed(1);
  console.log(`\n  ✅ Embedding 完成 (${embedTime}s, ${((memories.length / embedTime)).toFixed(0)} 条/s)`);
  
  // 创建表并插入
  const sampleEmbedding = allRecords[0].embedding;
  const table = await db.createTable(scope, [{
    id: 'init',
    text: 'init',
    embedding: sampleEmbedding
  }]);
  await table.delete("id = 'init'");
  await table.add(allRecords);
  
  console.log(`  ✅ 插入完成`);
  return table;
}

async function runScaleTest(db, scaleName, embeddingService) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 测试规模：${scaleName}`);
  console.log('='.repeat(70));
  
  // 加载数据
  const memories = JSON.parse(fs.readFileSync(path.join(config.dataDir, `memories_${scaleName}.json`), 'utf-8'));
  const queries = JSON.parse(fs.readFileSync(path.join(config.dataDir, `queries_${scaleName}.json`), 'utf-8'));
  
  console.log(`  📚 记忆数：${memories.length}`);
  console.log(`  ❓ 查询数：${queries.length}`);
  
  // 插入数据
  const table = await insertMemories(db, memories, config.testScope, embeddingService);
  
  // 批量生成查询 embedding
  console.log(`\n  🚀 生成查询 embedding...`);
  const queryStart = Date.now();
  const queryTexts = queries.map(q => q.query);
  const queryEmbeddings = await embeddingService.getEmbeddings(queryTexts);
  const queryTime = ((Date.now() - queryStart) / 1000).toFixed(1);
  console.log(`  ✅ 查询 embedding 完成 (${queryTime}s)`);
  
  // 运行检索测试
  console.log(`\n  🔍 运行检索测试...`);
  const metrics = new RecallMetrics();
  
  const searchStart = Date.now();
  for (let i = 0; i < queries.length; i++) {
    const query = queries[i];
    const targetId = query.target_memory_id;
    
    if (i % 100 === 0) {
      console.log(`  进度：${i + 1}/${queries.length}`);
    }
    
    const retrieved = await table.search(queryEmbeddings[i]).limit(10).toArray();
    metrics.record(query, targetId, retrieved);
  }
  const searchTime = ((Date.now() - searchStart) / 1000).toFixed(1);
  console.log(`  ✅ 检索完成 (${searchTime}s, ${(queries.length/searchTime).toFixed(0)} 查询/s)`);
  
  // 清理测试数据
  await db.dropTable(config.testScope);
  
  return metrics;
}

// ============ 主函数 ============

async function main() {
  console.log('🧪 Synthetic PerLTQA 召回率基准测试 (本地 GPU 加速)');
  console.log('🎮 GPU: AMD RX 6800 ROCm');
  console.log('📍 测试 Scope: 动态生成');
  console.log('='.repeat(70));
  
  const db = await lancedb.connect(config.dbPath);
  const embeddingService = new GPUEmbeddingService();
  
  try {
    console.log('\n⏳ 启动 GPU embedding 服务...');
    await embeddingService.start();
    console.log('✅ GPU 服务就绪\n');
    
    const allResults = {};
    const timings = {};
    
    for (const scale of SCALES) {
      const start = Date.now();
      const metrics = await runScaleTest(db, scale, embeddingService);
      const elapsed = ((Date.now() - start) / 1000).toFixed(1);
      
      allResults[scale] = metrics.getSummary();
      timings[scale] = elapsed;
      
      console.log(`\n📈 ${scale} 结果:`);
      console.log(`   Recall@5: ${allResults[scale].recall_at_5}`);
      console.log(`   Recall@10: ${allResults[scale].recall_at_10}`);
      console.log(`   MRR: ${allResults[scale].mrr}`);
      console.log(`   ⏱️  耗时：${elapsed}s`);
    }
    
    // 保存结果
    const report = {
      timestamp: new Date().toISOString(),
      gpu: 'AMD RX 6800 ROCm',
      model: 'BAAI/bge-m3',
      scales: allResults,
      timings: timings,
      data_points: SCALES.map(s => {
        const stats = JSON.parse(fs.readFileSync(path.join(config.dataDir, `stats_${s}.json`), 'utf-8'));
        return stats.num_memories;
      })
    };
    
    const outputPath = path.join(config.dataDir, 'benchmark_result_gpu.json');
    fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
    
    // 汇总报告
    console.log('\n' + '='.repeat(70));
    console.log('✅ 测试完成！');
    console.log('📄 报告已保存:', outputPath);
    
    console.log('\n📊 完整结果汇总:');
    console.log('\n' + '='.repeat(70));
    console.log(`| 规模 | 记忆数 | 查询数 | Recall@5 | Recall@10 | MRR | 耗时 |`);
    console.log(`|------|--------|--------|----------|-----------|-----|------|`);
    
    for (const scale of SCALES) {
      const stats = JSON.parse(fs.readFileSync(path.join(config.dataDir, `stats_${s}.json`), 'utf-8'));
      const r = allResults[scale];
      const t = timings[scale];
      console.log(`| ${scale} | ${stats.num_memories} | ${stats.num_queries} | ${r.recall_at_5} | ${r.recall_at_10} | ${r.mrr} | ${t}s |`);
    }
    
    console.log('='.repeat(70));
    
    // 绘制曲线数据
    console.log('\n📈 绘图数据 (CSV 格式):');
    console.log('scale,num_memories,recall_at_5,recall_at_10,mrr,time_seconds');
    for (const scale of SCALES) {
      const stats = JSON.parse(fs.readFileSync(path.join(config.dataDir, `stats_${scale}.json`), 'utf-8'));
      const r = allResults[scale];
      console.log(`${scale},${stats.num_memories},${r.recall_at_5},${r.recall_at_10},${r.mrr},${timings[scale]}`);
    }
    
  } finally {
    await embeddingService.stop();
  }
}

// 运行
main().catch(console.error);
