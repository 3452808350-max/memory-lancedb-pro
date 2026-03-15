#!/usr/bin/env node
/**
 * 导入错误归因记忆到 LanceDB
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

async function importErrorMemory() {
  console.log('🚨 导入错误归因记忆到 LanceDB...\n');
  
  // 读取错误记忆
  const errorData = JSON.parse(
    fs.readFileSync('/home/kyj/.openclaw/workspace/error_20260313_001.json', 'utf-8')
  );
  
  const memory = errorData.error_memory;
  
  console.log(`📝 错误 ID: ${memory.metadata.error_id}`);
  console.log(`📋 错误类型：${memory.metadata.error_type}`);
  console.log(`⚠️  严重程度：${memory.metadata.severity}`);
  console.log(`🔧 根因分类：${memory.metadata.root_cause_category}`);
  console.log();
  
  // 连接数据库
  const db = await lancedb.connect(config.dbPath);
  const table = await db.openTable('memories');
  
  try {
    // 生成向量
    const embedding = await getEmbedding(memory.text);
    
    // 准备记录
    const record = {
      id: memory.metadata.error_id,
      text: memory.text,
      vector: embedding,
      category: memory.category,
      scope: memory.scope,
      importance: memory.importance,
      timestamp: Date.now(),
      metadata: JSON.stringify(memory.metadata)
    };
    
    // 检查是否已存在
    const existing = await table.query(`SELECT id FROM memories WHERE id = '${memory.metadata.error_id}'`).toArray();
    
    if (existing.length > 0) {
      console.log('⚠️  该错误记忆已存在，更新中...\n');
      await table.delete(`id = '${memory.metadata.error_id}'`);
    }
    
    // 插入
    await table.add([record]);
    
    console.log('✅ 错误归因记忆导入成功！\n');
    console.log('='.repeat(70));
    console.log('\n📊 记忆详情:\n');
    console.log(`   错误 ID: ${memory.metadata.error_id}`);
    console.log(`   错误描述：${memory.text.slice(0, 100)}...`);
    console.log(`   影响系统：${memory.metadata.affected_system}`);
    console.log(`   根因分类：${memory.metadata.root_cause_category}`);
    console.log(`   解决时间：${memory.metadata.resolution_time_minutes} 分钟`);
    console.log(`   标签：${memory.tags.join(', ')}`);
    console.log(`   经验教训：${memory.metadata.lessons.length} 条`);
    console.log();
    console.log('🔍 未来可通过以下方式检索:');
    console.log(`   - 错误类型：${memory.metadata.error_type}`);
    console.log(`   - 影响系统：${memory.metadata.affected_system}`);
    console.log(`   - 根因分类：${memory.metadata.root_cause_category}`);
    console.log(`   - 标签：${memory.tags.join(', ')}`);
    console.log();
    console.log('='.repeat(70));
    
  } catch (error) {
    console.log(`❌ 导入失败：${error.message}\n`);
    throw error;
  }
}

importErrorMemory().catch(console.error);
