#!/usr/bin/env node
/**
 * 导入补充记忆到 LanceDB
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

async function importMemories() {
  console.log('📥 开始导入补充记忆到 LanceDB...\n');
  
  // 读取补充记忆
  const memories = JSON.parse(
    fs.readFileSync('/home/kyj/.openclaw/workspace/memory_supplement.json', 'utf-8')
  );
  
  console.log(`📝 待导入记忆：${memories.length} 条\n`);
  
  // 连接数据库
  const db = await lancedb.connect(config.dbPath);
  const table = await db.openTable('memories');
  
  let success = 0;
  let failed = 0;
  
  for (let i = 0; i < memories.length; i++) {
    const mem = memories[i];
    
    try {
      console.log(`[${i + 1}/${memories.length}] 导入：${mem.text.slice(0, 50)}...`);
      
      // 生成向量
      const embedding = await getEmbedding(mem.text);
      
      // 准备记录
      const record = {
        id: `supplement-${Date.now()}-${i}`,
        text: mem.text,
        vector: embedding,
        category: mem.category,
        scope: mem.scope || 'global',
        importance: mem.importance,
        timestamp: Date.now(),
        metadata: JSON.stringify({
          ...mem.metadata,
          tags: mem.tags
        })
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
  console.log('\n📊 导入完成！\n');
  console.log(`   ✅ 成功：${success} 条`);
  console.log(`   ❌ 失败：${failed} 条`);
  console.log(`   📊 数据库总计：${finalCount} 条记忆\n`);
  console.log('='.repeat(70));
}

importMemories().catch(console.error);
