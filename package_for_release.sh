#!/bin/bash
# MemQ 开源打包脚本

set -e

echo "📦 MemQ 开源打包..."

# 项目根目录
PROJECT_ROOT="/home/kyj/.openclaw/workspace/memory-lancedb-pro"
RELEASE_NAME="memq-v1.0.0"
RELEASE_DIR="/tmp/$RELEASE_NAME"

# 清理旧目录
rm -rf $RELEASE_DIR
mkdir -p $RELEASE_DIR

# 复制必要文件
echo "复制源代码..."
rsync -av --exclude='.git' \
      --exclude='.venv' \
      --exclude='node_modules' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='results/*.log' \
      --exclude='results/*.json' \
      --exclude='memory_db/*.jsonl' \
      --exclude='.DS_Store' \
      $PROJECT_ROOT/ $RELEASE_DIR/

# 创建空目录占位符
mkdir -p $RELEASE_DIR/{results,memory_db}
touch $RELEASE_DIR/results/.gitkeep
touch $RELEASE_DIR/memory_db/.gitkeep

# 设置权限
chmod +x $RELEASE_DIR/scripts/*.py 2>/dev/null || true

# 创建压缩包
echo "创建压缩包..."
cd /tmp
tar -czf $RELEASE_NAME.tar.gz $RELEASE_NAME
rm -rf $RELEASE_DIR

echo ""
echo "✅ 打包完成！"
echo "📦 位置：/tmp/$RELEASE_NAME.tar.gz"
echo "📊 大小：$(du -h /tmp/$RELEASE_NAME.tar.gz | cut -f1)"
