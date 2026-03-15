#!/usr/bin/env node
/**
 * Synthetic PerLTQA 召回率基准测试 - 快速版 (原生 HTTP)
 * 使用阿里云 Embedding API + 直接 ID 匹配
 */

import * as lancedb from '@lancedb/lancedb';
import fs from 'fs';
import path from 'path';

// ============ 配置 ============
const config = {
  dbPath: '/home/kyj/.openclaw/workspace/lancedb',
  apiKey: 'sk-e8b53592ebe841f28a03d4d54024761c',
  dataDir: '/home/kyj/.openclaw/workspace/synthetic_perltqa'
};

const SCALES = ['baseline', 'small', 'medium', 'medium-large', 'large'];

// ============ Embedding 服务 (用 child_process 调用 curl) ============

async function getEmbedding(text) {
  const { exec } = await import('child_process');
  const util = await import('util');
  const execPromise = util.promisify(exec);
  
  // 使用 Qwen3-Embedding-8B
  const cmd = `curl -s -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings \\
    -H "Authorization: Bearer ${config.apiKey}" \\
    -H "Content-Type: application/json" \\
    -d '${JSON.stringify({ model: 'text-embedding-v2', input: [text] })}'`;
  
  try {
    const { stdout } = await execPromise(cmd, { timeout: 30000 });
    const result = JSON.parse(stdout);
    if (result.data && result.data[0] && result.data[0].embedding) {
      return result.data[0].embedding;
    } else {
      throw new Error('Invalid response: ' + stdout);
    }
  } catch (e) {
    throw e;
  }
}

async function getEmbeddings(texts) {
  const embeddings = [];
  for (let i = 0; i < texts.length; i++) {
    if (i % 50 === 0) {
      process.stdout.write(`\r  进度：${i+1}/${texts.length}`);
    }
    const emb = await getEmbedding(texts[i]);
    embeddings.push(emb);
  }
  console.log(`\r  进度：${texts.length}/${texts.length}`);
  return embeddings;
}

// ============ 测试逻辑 ============

class RecallMetrics {
  constructor() { this.results = []; }
  
  record(query, targetId, retrieved) {
    const hit = retrieved.slice(0, 5).some(r => String(r.id) === String(targetId));
    const rank = retrieved.findIndex(r => String(r.id) === String(targetId)) + 1;
    this.results.push({ query_id: query.id, target_id: targetId, hit_at_5: hit, rank: rank > 0 ? rank : -1 });
  }
  
  getRecallAtK(k = 5) {
    return this.results.filter(r => r.hit_at_5).length / this.results.length;
  }
  
  getMRR() {
    let sum = 0;
    for (const r of this.results) if (r.rank > 0) sum += 1 / r.rank;
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

async function runScaleTest(db, scaleName) {
  console.log(`\n${'='.repeat(70)}`);
  console.log(`📊 测试规模：${scaleName}`);
  console.log('='.repeat(70));
  
  const memories = JSON.parse(fs.readFileSync(path.join(config.dataDir, `memories_${scaleName}.json`), 'utf-8'));
  const queries = JSON.parse(fs.readFileSync(path.join(config.dataDir, `queries_${scaleName}.json`), 'utf-8'));
  
  console.log(`  📚 记忆数：${memories.length}`);
  console.log(`  ❓ 查询数：${queries.length}`);
  
  const scope = `perltqa-fast-${Date.now()}`;
  
  // 生成记忆 embedding
  console.log(`\n  🚀 生成记忆 embedding...`);
  const embedStart = Date.now();
  const memoryEmbeds = await getEmbeddings(memories.map(m => m.content));
  const embedTime = ((Date.now() - embedStart) / 1000).toFixed(1);
  console.log(`  ✅ 记忆 embedding 完成 (${embedTime}s, ${(memories.length/embedTime).toFixed(0)} 条/s)`);
  
  // 创建表
  try { await db.dropTable(scope); } catch(e) {}
  
  const table = await db.createTable(scope, [{ id: 'init', text: 'init', embedding: memoryEmbeds[0] }]);
  await table.delete("id = 'init'");
  
  // 批量插入
  const batchSize = 500;
  for (let i = 0; i < memories.length; i += batchSize) {
    const batch = memories.slice(i, i + batchSize);
    const embeds = memoryEmbeds.slice(i, i + batchSize);
    await table.add(batch.map((m, idx) => ({ id: m.id, text: m.content, embedding: embeds[idx] })));
  }
  console.log(`  ✅ 插入完成\n`);
  
  // 生成查询 embedding
  console.log(`  🚀 生成查询 embedding...`);
  const queryEmbeds = await getEmbeddings(queries.map(q => q.query));
  console.log(`  ✅ 查询 embedding 完成\n`);
  
  // 运行检索
  console.log(`  🔍 运行检索...`);
  const metrics = new RecallMetrics();
  const searchStart = Date.now();
  
  for (let i = 0; i < queries.length; i++) {
    if (i % 100 === 0) console.log(`  进度：${i+1}/${queries.length}`);
    const retrieved = await table.search(queryEmbeds[i]).limit(10).toArray();
    metrics.record(queries[i], queries[i].target_memory_id, retrieved);
  }
  
  const searchTime = ((Date.now() - searchStart) / 1000).toFixed(1);
  console.log(`  ✅ 检索完成 (${searchTime}s, ${(queries.length/searchTime).toFixed(0)} 查询/s)`);
  
  await db.dropTable(scope);
  return [metrics, embedTime + searchTime];
}

// ============ 主函数 ============

async function main() {
  console.log('🧪 Synthetic PerLTQA 召回率基准测试 (快速 API 版)');
  console.log('📦 模型：阿里云 text-embedding-v3');
  console.log('⚡ 优化：跳过 LLM 评估，直接 ID 匹配');
  console.log('='.repeat(70));
  
  const db = await lancedb.connect(config.dbPath);
  const allResults = {};
  const allTimings = {};
  
  for (const scale of SCALES) {
    const start = Date.now();
    const [metrics, _] = await runScaleTest(db, scale);
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    
    allResults[scale] = metrics.getSummary();
    allTimings[scale] = elapsed;
    
    const r = allResults[scale];
    console.log(`\n📈 ${scale} 结果:`);
    console.log(`   Recall@5:  ${r.recall_at_5 || r.recall_at_5}`);
    console.log(`   Recall@10: ${r.recall_at_10 || r.recall_at_10}`);
    console.log(`   MRR:       ${r.mrr || r.mrr}`);
    console.log(`   ⏱️  总耗时：${elapsed}s`);
  }
  
  // 保存报告
  const report = {
    timestamp: new Date().toISOString(),
    model: 'aliyun/text-embedding-v3',
    results: allResults,
    timings: allTimings
  };
  
  const outputPath = path.join(config.dataDir, 'benchmark_result_fast.json');
  fs.writeFileSync(outputPath, JSON.stringify(report, null, 2));
  
  // 汇总
  console.log('\n' + '='.repeat(70));
  console.log('✅ 测试完成！');
  console.log('\n📊 完整结果:');
  console.log('='.repeat(70));
  console.log(`| 规模 | 记忆数 | 查询数 | Recall@5 | Recall@10 | MRR | 耗时 |`);
  console.log(`|------|--------|--------|----------|-----------|-----|------|`);
  
  for (const scale of SCALES) {
    const stats = JSON.parse(fs.readFileSync(path.join(config.dataDir, `stats_${scale}.json`), 'utf-8'));
    const r = allResults[scale];
    const t = allTimings[scale];
    console.log(`| ${scale} | ${stats.num_memories} | ${stats.num_queries} | ${r.recall_at_5} | ${r.recall_at_10} | ${r.mrr} | ${t}s |`);
  }
  console.log('='.repeat(70));
}

main().catch(console.error);
