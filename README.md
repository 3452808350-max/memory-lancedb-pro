# 🧠 memory-lancedb-pro

> **OpenClaw 增强型 LanceDB 长期记忆插件**

混合检索（Vector + BM25）· 跨编码器 Rerank · 多 Scope 隔离 · 管理 CLI

[![OpenClaw Plugin](https://img.shields.io/badge/OpenClaw-Plugin-blue)](https://github.com/openclaw/openclaw)
[![LanceDB](https://img.shields.io/badge/LanceDB-Vectorstore-orange)](https://lancedb.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## ✨ 特性

- 🔍 **混合检索**: Vector + BM25 全文检索，RRF 融合排序
- 🔄 **Cross-Encoder Rerank**: Jina Reranker 重排序
- 📊 **时效性加成**: 时间衰减、重要性权重
- 🎯 **多 Scope 隔离**: global / agent / project / user
- 🧹 **噪声过滤**: 自动过滤低质量记忆
- 🛠 **管理 CLI**: list / search / stats / delete / export

---

## 🚀 快速开始

### 1. 安装

```bash
cd ~/.openclaw/extensions
git clone https://github.com/3452808350-max/memory-lancedb-pro.git
cd memory-lancedb-pro
npm install
```

### 2. 配置

在 `~/.openclaw/openclaw.json` 中添加：

```json
{
  "plugins": {
    "allow": ["memory-lancedb-pro"],
    "entries": {
      "memory-lancedb-pro": {
        "enabled": true,
        "config": {
          "embedding": {
            "provider": "openai-compatible",
            "apiKey": "sk-xxx",
            "model": "text-embedding-v3",
            "baseURL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "dimensions": 1024
          },
          "dbPath": "/home/kyj/.openclaw/workspace/lancedb"
        }
      }
    }
  }
}
```

### 3. 重启

```bash
openclaw gateway restart
```

---

## 📖 使用

### Agent 工具

```python
# 存储记忆
memory_store(text="K 偏好 TypeScript", category="preference", importance=0.8)

# 检索记忆
memory_recall(query="编程语言偏好", limit=5)
```

### CLI 命令

```bash
openclaw memory list              # 列出记忆
openclaw memory search "关键词"    # 搜索
openclaw memory stats             # 统计
openclaw memory export            # 导出
```

---

## 🏗 架构

```
index.ts → store.ts → LanceDB
         → retriever.ts → Vector + BM25
         → tools.ts → Agent 工具
         → cli.ts → CLI 命令
```

详细开发文档见 [DEVELOPMENT.md](DEVELOPMENT.md)

---

## 📄 License

MIT License

---

*版本：2026.2.16*
