#!/usr/bin/env node
/**
 * memory-lancedb-pro 记忆召回率测试
 * 使用 Kimi K2.5 作为评估 Agent
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
  },
  kimi: {
    apiKey: 'sk-e8b53592ebe841f28a03d4d54024761c',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'kimi-k2.5'
  }
};

// 测试查询集（基于已存储的记忆）
const testQueries = [
  {
    query: "K 的性格特点是什么？",
    expected: ["专注", "好奇", "独立"],
    category: "preference"
  },
  {
    query: "K 的价值观是什么？",
    expected: ["自由", "平等"],
    category: "preference"
  },
  {
    query: "K 有什么特长？",
    expected: ["技术", "摄影", "HiFi", "电脑硬件"],
    category: "preference"
  },
  {
    query: "memory-lancedb-pro 项目的检索率提升了多少？",
    expected: ["68%", "78%", "15%"],
    category: "fact"
  },
  {
    query: "K 申请的是哪所韩国大学？",
    expected: ["成均馆", "Sungkyunkwan"],
    category: "fact"
  },
  {
    query: "K 的技术实践经历有哪些？",
    expected: ["OpenWrt", "Linux", "VPS", "Ollama", "vLLM"],
    category: "fact"
  },
  {
    query: "K 的母亲对 K 有什么影响？",
    expected: ["西方教育", "自由", "平等", "开明"],
    category: "fact"
  },
  {
    query: "K 毕业后想做什么？",
    expected: ["三星", "Naver", "AI 工程师"],
    category: "decision"
  },
  {
    query: "K 遇到过什么困难？",
    expected: ["AI Brain Fry", "FOMO", "焦虑"],
    category: "fact"
  },
  {
    query: "K 的学习计划是什么？",
    expected: ["大一", "大二", "大三", "大四", "实习"],
    category: "decision"
  }
];

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

async function queryKimi(prompt) {
  const client = new OpenAI({
    apiKey: config.kimi.apiKey,
    baseURL: config.kimi.baseURL
  });
  
  const response = await client.chat.completions.create({
    model: config.kimi.model,
    messages: [
      {
        role: "system",
        content: "你是一个记忆召回率评估助手。我会给你一个问题、检索到的记忆、以及期望的关键信息。请判断检索到的记忆是否包含期望的信息。"
      },
      {
        role: "user",
        content: prompt
      }
    ],
    temperature: 0.1,
    max_tokens: 500
  });
  
  return response.choices[0].message.content;
}

async function runTest() {
  console.log('🧪 memory-lancedb-pro 记忆召回率测试\n');
  console.log('🤖 评估模型：Kimi K2.5\n');
  console.log('='.repeat(70));
  
  const db = await lancedb.connect(config.dbPath);
  const table = await db.openTable('memories');
  
  const total = testQueries.length;
  let success = 0;
  let partial = 0;
  let failed = 0;
  
  const results = [];
  
  for (let i = 0; i < testQueries.length; i++) {
    const { query, expected, category } = testQueries[i];
    
    console.log(`\n[测试 ${i + 1}/${total}]`);
    console.log(`查询：${query}`);
    console.log(`类别：${category}`);
    console.log(`期望关键词：${expected.join(', ')}`);
    
    // 向量检索
    const embedding = await getEmbedding(query);
    const retrieved = await table.search(embedding).limit(5).toArray();
    
    console.log(`\n检索到 ${retrieved.length} 条记忆:`);
    retrieved.forEach((r, j) => {
      console.log(`  ${j + 1}. [${r.category}] ${r.text.slice(0, 100)}... (${r.importance})`);
    });
    
    // 使用 Kimi 评估
    const evalPrompt = `
请评估以下记忆检索结果：

问题：${query}
期望包含的关键信息：${expected.join(', ')}

检索到的记忆：
${retrieved.map((r, j) => `${j + 1}. ${r.text}`).join('\n')}

请回答：
1. 检索结果是否包含所有期望的关键信息？（是/部分/否）
2. 如果部分包含，缺少哪些？
3. 召回率评分（0-100 分）

请简洁回答。
`;
    
    const evalResult = await queryKimi(evalPrompt);
    console.log(`\nKimi 评估:\n${evalResult}`);
    
    // 解析评估结果
    const isComplete = evalResult.includes('是') && !evalResult.includes('部分');
    const isPartial = evalResult.includes('部分');
    const scoreMatch = evalResult.match(/(\d+)\s*分/);
    const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;
    
    if (isComplete || score >= 80) {
      success++;
      console.log('✅ 成功');
    } else if (isPartial || score >= 50) {
      partial++;
      console.log('⚠️  部分成功');
    } else {
      failed++;
      console.log('❌ 失败');
    }
    
    results.push({
      query,
      category,
      expected,
      retrieved_count: retrieved.length,
      success: isComplete,
      partial: isPartial,
      score,
      eval: evalResult
    });
    
    console.log('─'.repeat(70));
  }
  
  // 汇总结果
  console.log('\n\n' + '='.repeat(70));
  console.log('📊 测试结果汇总\n');
  
  console.log(`总查询数：${total}`);
  console.log(`✅ 完全召回：${success} (${(success/total*100).toFixed(1)}%)`);
  console.log(`⚠️  部分召回：${partial} (${(partial/total*100).toFixed(1)}%)`);
  console.log(`❌ 召回失败：${failed} (${(failed/total*100).toFixed(1)}%)`);
  console.log(`\n综合召回率：${((success + partial * 0.5) / total * 100).toFixed(1)}%`);
  
  // 按类别统计
  console.log('\n按类别统计:');
  const byCategory = {};
  for (const r of results) {
    if (!byCategory[r.category]) {
      byCategory[r.category] = { total: 0, success: 0, partial: 0 };
    }
    byCategory[r.category].total++;
    if (r.success) byCategory[r.category].success++;
    if (r.partial) byCategory[r.category].partial++;
  }
  
  for (const [cat, stats] of Object.entries(byCategory)) {
    console.log(`  ${cat}: ${stats.success}/${stats.total} (${(stats.success/stats.total*100).toFixed(1)}%)`);
  }
  
  // 保存结果
  const fs = await import('fs');
  const report = {
    timestamp: new Date().toISOString(),
    model: 'kimi-k2.5',
    total,
    success,
    partial,
    failed,
    recall_rate: ((success + partial * 0.5) / total * 100).toFixed(1),
    by_category: byCategory,
    details: results
  };
  
  fs.writeFileSync(
    '/home/kyj/.openclaw/workspace/memory-lancedb-pro/eval/recall_test_result.json',
    JSON.stringify(report, null, 2)
  );
  
  console.log('\n✅ 测试结果已保存到：eval/recall_test_result.json\n');
}

// 运行测试
runTest().catch(console.error);
