#!/usr/bin/env node
/**
 * memory-lancedb-pro 性能评估脚本
 * 
 * 测试混合检索相比单一检索的准确率提升
 */

import * as lancedb from '@lancedb/lancedb';
import { OpenAI } from 'openai';

// 配置
const config = {
  dbPath: '/home/kyj/.openclaw/workspace/lancedb',
  embedding: {
    apiKey: 'sk-e8b53592ebe841f28a03d4d54024761c',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'text-embedding-v3'
  }
};

// 测试查询集（100 个典型查询）
const testQueries = [
  { query: "TypeScript 偏好", expected: ["TypeScript", "类型安全", "JavaScript"] },
  { query: "Python 缺点", expected: ["动态类型", "运行错误"] },
  { query: "Linux 发行版", expected: ["Debian", "Arch", "Red Hat"] },
  { query: "AI 模型部署", expected: ["Ollama", "vLLM", "本地部署"] },
  { query: "网络协议", expected: ["HTTP", "SOCKS", "OSI"] },
  // ... 可以扩展到 100 个查询
];

// 评估指标
class Metrics {
  constructor() {
    this.tp = 0; // True Positives
    this.fp = 0; // False Positives
    this.fn = 0; // False Negatives
    this.reciprocalRanks = [];
  }

  recallAtK(k, retrieved, expected) {
    const retrievedSet = new Set(retrieved.slice(0, k).map(r => r.text.toLowerCase()));
    const expectedSet = new Set(expected.map(e => e.toLowerCase()));
    
    let hits = 0;
    for (const exp of expectedSet) {
      if (retrievedSet.has(exp)) hits++;
    }
    
    return hits / expectedSet.size;
  }

  mrr(retrieved, expected) {
    const expectedSet = new Set(expected.map(e => e.toLowerCase()));
    
    for (let i = 0; i < retrieved.length; i++) {
      if (expectedSet.has(retrieved[i].text.toLowerCase())) {
        this.reciprocalRanks.push(1 / (i + 1));
        return 1 / (i + 1);
      }
    }
    
    this.reciprocalRanks.push(0);
    return 0;
  }

  averageMRR() {
    if (this.reciprocalRanks.length === 0) return 0;
    return this.reciprocalRanks.reduce((a, b) => a + b, 0) / this.reciprocalRanks.length;
  }
}

async function getEmbedding(text) {
  const client = new OpenAI({
    apiKey: config.embedding.apiKey,
    baseURL: config.embedding.baseURL
  });
  
  const response = await client.embeddings.create({
    model: config.embedding.model,
    input: text
  });
  
  return response.data[0].embedding;
}

async function vectorSearch(table, query, limit = 10) {
  const embedding = await getEmbedding(query);
  return await table.search(embedding).limit(limit).toArray();
}

async function bm25Search(table, query, limit = 10) {
  // LanceDB BM25 搜索（简化版）
  const results = await table.query(`SELECT * FROM memories WHERE text LIKE '%${query}%' LIMIT ${limit}`).toArray();
  return results.map(r => ({ ...r, score: 0.5 })); // 简化评分
}

function rrfFusion(vectorResults, bm25Results, k = 60) {
  const scores = new Map();
  
  // Vector scores
  vectorResults.forEach((doc, i) => {
    const score = scores.get(doc.id) || 0;
    scores.set(doc.id, score + 1 / (k + i));
  });
  
  // BM25 scores
  bm25Results.forEach((doc, i) => {
    const score = scores.get(doc.id) || 0;
    scores.set(doc.id, score + 1 / (k + i));
  });
  
  // Sort by RRF score
  const sorted = Array.from(scores.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10);
  
  return sorted.map(([id, score]) => ({
    id,
    score,
    text: vectorResults.find(r => r.id === id)?.text || bm25Results.find(r => r.id === id)?.text
  }));
}

async function runEvaluation() {
  console.log('🧪 开始性能评估...\n');
  console.log('='.repeat(60));
  
  const db = await lancedb.connect(config.dbPath);
  const table = await db.openTable('memories');
  
  const metrics = {
    vector: new Metrics(),
    bm25: new Metrics(),
    hybrid: new Metrics()
  };
  
  console.log(`\n📊 测试查询数：${testQueries.length}\n`);
  
  for (const { query, expected } of testQueries) {
    console.log(`查询：${query}`);
    
    // Vector search
    const vectorResults = await vectorSearch(table, query);
    const vectorR5 = metrics.vector.recallAtK(5, vectorResults, expected);
    const vectorMRR = metrics.vector.mrr(vectorResults, expected);
    
    // BM25 search
    const bm25Results = await bm25Search(table, query);
    const bm25R5 = metrics.bm25.recallAtK(5, bm25Results, expected);
    const bm25MRR = metrics.bm25.mrr(bm25Results, expected);
    
    // Hybrid search
    const hybridResults = rrfFusion(vectorResults, bm25Results);
    const hybridR5 = metrics.hybrid.recallAtK(5, hybridResults, expected);
    const hybridMRR = metrics.hybrid.mrr(hybridResults, expected);
    
    console.log(`  Vector: R@5=${(vectorR5 * 100).toFixed(0)}%, MRR=${vectorMRR.toFixed(2)}`);
    console.log(`  BM25:   R@5=${(bm25R5 * 100).toFixed(0)}%, MRR=${bm25MRR.toFixed(2)}`);
    console.log(`  Hybrid: R@5=${(hybridR5 * 100).toFixed(0)}%, MRR=${hybridMRR.toFixed(2)}`);
    console.log();
  }
  
  // 汇总结果
  console.log('='.repeat(60));
  console.log('\n📊 汇总结果:\n');
  
  console.log('| 方法 | Recall@5 | Recall@10 | MRR |');
  console.log('|------|----------|-----------|-----|');
  console.log(`| Vector | ${(metrics.vector.recallAtK(5, [], []) * 100).toFixed(0)}% | - | ${metrics.vector.averageMRR().toFixed(2)} |`);
  console.log(`| BM25 | ${(metrics.bm25.recallAtK(5, [], []) * 100).toFixed(0)}% | - | ${metrics.bm25.averageMRR().toFixed(2)} |`);
  console.log(`| **Hybrid** | **${(metrics.hybrid.recallAtK(5, [], []) * 100).toFixed(0)}%** | - | **${metrics.hybrid.averageMRR().toFixed(2)}** |`);
  
  console.log('\n✅ 评估完成！\n');
}

// 运行评估
runEvaluation().catch(console.error);
