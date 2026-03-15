#!/usr/bin/env node
/**
 * 导入 DSS 开发日志到 LanceDB
 */

import * as lancedb from '@lancedb/lancedb';
import { OpenAI } from 'openai';
import fs from 'fs';

const config = {
  dbPath: '/home/kyj/.openclaw/workspace/lancedb',
  embedding: {
    apiKey: 'sk-e8b53592ebe841f28a03d4d54024761c',
    baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model: 'text-embedding-v3'
  }
};

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

async function importDSSLogs() {
  console.log('📊 导入 DSS 开发日志到 LanceDB...\n');
  
  // DSS 开发日志
  const dssLogs = [
    {
      text: "DSS 日志系统实现：1.数据溯源日志 - 记录所有数据真实来源 URL，标注成功/失败/缓存/不可用状态；2.参数权重日志 - 记录初始权重配置、每次参数调整（原因、影响）、最优解搜索过程；3.生成审计报告。严格执行数据原则：不允许伪造数据，所有数据标注真实来源 URL。",
      category: "fact",
      scope: "global",
      importance: 0.9,
      tags: ["DSS", "日志系统", "数据溯源", "参数权重"],
      metadata: {
        source: "DSS_LOGGER_SUMMARY.md",
        type: "开发日志",
        system: "DSS",
        feature: "日志系统"
      }
    },
    {
      text: "DSS 数据原则：不允许伪造数据。旧代码 generate_simulated_data 已删除，新代码明确标注数据不可用状态。所有数据源标注真实 URL：Alpha Vantage、AKShare、Sina API、本地缓存。记录所有参数调整和权重变化，确保可追溯、可审计、可复现。",
      category: "decision",
      scope: "global",
      importance: 0.95,
      tags: ["DSS", "数据原则", "不伪造", "可追溯"],
      metadata: {
        source: "DSS_LOGGER_SUMMARY.md",
        type: "开发日志",
        system: "DSS",
        principle: "数据真实性"
      }
    },
    {
      text: "DSS 项目拉取：Microsoft Qlib (/home/kyj/qlib, 11k stars) - AI 量化投资平台，学习重点：因子库、回测引擎。Microsoft RD-Agent (/home/kyj/RD-Agent) - NeurIPS 2025 接收，MLE-bench 第 1 名，学习重点：自动因子挖掘、多 Agent 协作。",
      category: "fact",
      scope: "global",
      importance: 0.85,
      tags: ["DSS", "Qlib", "RD-Agent", "项目"],
      metadata: {
        source: "DSS_LOGGER_SUMMARY.md",
        type: "开发日志",
        system: "DSS",
        project: "外部项目"
      }
    },
    {
      text: "DSS 文件结构：dss_logger.py（日志系统核心）、dss_v2.py（DSS 主程序，已集成日志）、test_dss_logger.py（测试脚本）、data_logs/（数据源日志目录）、weight_logs/（参数权重日志目录）。日志格式：JSONL 可追溯格式。",
      category: "fact",
      scope: "global",
      importance: 0.8,
      tags: ["DSS", "文件结构", "日志格式"],
      metadata: {
        source: "DSS_LOGGER_SUMMARY.md",
        type: "开发日志",
        system: "DSS",
        feature: "文件结构"
      }
    },
    {
      text: "DSS 优化计划：1.集成 Qlib 因子库；2.实现 Agent 辅助参数优化；3.添加交易解释生成；4.学习 RD-Agent 自动因子挖掘。目标：提升 DSS 预测准确性和可解释性。",
      category: "decision",
      scope: "global",
      importance: 0.85,
      tags: ["DSS", "优化计划", "Qlib", "RD-Agent"],
      metadata: {
        source: "DSS_OPTIMIZATION_PLAN.md",
        type: "开发日志",
        system: "DSS",
        feature: "优化计划"
      }
    }
  ];
  
  console.log(`📝 待导入 DSS 日志：${dssLogs.length} 条\n`);
  
  // 连接数据库
  const db = await lancedb.connect(config.dbPath);
  const table = await db.openTable('memories');
  
  let success = 0;
  let failed = 0;
  
  for (let i = 0; i < dssLogs.length; i++) {
    const log = dssLogs[i];
    
    try {
      console.log(`[${i + 1}/${dssLogs.length}] 导入：${log.text.slice(0, 50)}...`);
      
      // 生成向量
      const embedding = await getEmbedding(log.text);
      
      // 准备记录
      const record = {
        id: `DSS-LOG-${Date.now()}-${i}`,
        text: log.text,
        vector: embedding,
        category: log.category,
        scope: log.scope,
        importance: log.importance,
        timestamp: Date.now(),
        metadata: JSON.stringify(log.metadata)
      };
      
      // 插入
      await table.add([record]);
      console.log(`   ✅ 成功\n`);
      success++;
      
    } catch (error) {
      console.log(`   ❌ 失败：${error.message}\n`);
      failed++;
    }
  }
  
  // 统计
  const finalCount = await table.countRows();
  
  console.log('='.repeat(70));
  console.log('\n📊 DSS 日志导入完成！\n');
  console.log(`   ✅ 成功：${success} 条`);
  console.log(`   ❌ 失败：${failed} 条`);
  console.log(`   📊 数据库总计：${finalCount} 条记忆`);
  console.log('\n='.repeat(70));
}

importDSSLogs().catch(console.error);
