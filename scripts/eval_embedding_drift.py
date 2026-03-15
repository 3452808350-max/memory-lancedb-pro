#!/usr/bin/env python3
"""
Embedding Drift Detection

检测 embedding 分布变化。

Usage:
    python scripts/eval_embedding_drift.py --memory memory_db/memories.jsonl
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

def load_memories(path):
    memories = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    return memories

def main():
    parser = argparse.ArgumentParser(description='Embedding Drift Detection')
    parser.add_argument('--memory', required=True, help='Path to memories.jsonl')
    parser.add_argument('--model', default='BAAI/bge-m3', help='Embedding model')
    parser.add_argument('--batch_size', type=int, default=64, help='Embedding batch size')
    args = parser.parse_args()
    
    print(f"📊 Loading memories from {args.memory}...")
    memories = load_memories(args.memory)
    print(f"✅ Loaded {len(memories)} memories")
    
    # Sort by timestamp if available
    try:
        memories.sort(key=lambda x: x.get('timestamp', ''))
    except:
        pass
    
    # Load model
    print(f"\n🤖 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Generate embeddings in batches
    print("\n🚀 Generating embeddings...")
    texts = [m.get('text', '') for m in memories]
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    
    # Compute statistics
    norms = np.linalg.norm(embeddings, axis=1)
    centroid = embeddings.mean(axis=0)
    
    # Split into old/new for drift detection
    mid = len(embeddings) // 2
    old_emb = embeddings[:mid]
    new_emb = embeddings[mid:]
    
    old_centroid = old_emb.mean(axis=0)
    new_centroid = new_emb.mean(axis=0)
    
    # Drift metrics
    centroid_shift = np.linalg.norm(old_centroid - new_centroid)
    centroid_cosine = np.dot(old_centroid, new_centroid) / (np.linalg.norm(old_centroid) * np.linalg.norm(new_centroid))
    
    norm_change = new_emb.mean() - old_emb.mean()
    variance_change = new_emb.var() - old_emb.var()
    
    # Report
    print("\n" + "="*60)
    print("📈 Embedding Drift Detection Results")
    print("="*60)
    print(f"Total Embeddings: {len(embeddings)}")
    print(f"Old Set: {len(old_emb)}, New Set: {len(new_emb)}")
    print(f"\n📊 Global Statistics:")
    print(f"  Mean Norm: {norms.mean():.4f}")
    print(f"  Std Norm: {norms.std():.4f}")
    print(f"  Centroid Norm: {np.linalg.norm(centroid):.4f}")
    print(f"\n📉 Drift Metrics:")
    print(f"  Centroid Shift (L2): {centroid_shift:.4f}")
    print(f"  Centroid Cosine Sim: {centroid_cosine:.4f}")
    print(f"  Norm Change: {norm_change:.4f}")
    print(f"  Variance Change: {variance_change:.4f}")
    
    # Threshold check
    print(f"\n⚠️  Drift Assessment:")
    if centroid_cosine < 0.95:
        print(f"  🔴 HIGH DRIFT detected (cosine < 0.95)")
    elif centroid_cosine < 0.98:
        print(f"  🟡 MODERATE DRIFT detected (cosine < 0.98)")
    else:
        print(f"  🟢 LOW DRIFT (cosine >= 0.98)")
    
    # Save results
    results = {
        'total_embeddings': len(embeddings),
        'mean_norm': float(norms.mean()),
        'std_norm': float(norms.std()),
        'centroid_norm': float(np.linalg.norm(centroid)),
        'centroid_shift_l2': float(centroid_shift),
        'centroid_cosine_sim': float(centroid_cosine),
        'norm_change': float(norm_change),
        'variance_change': float(variance_change),
        'drift_assessment': 'HIGH' if centroid_cosine < 0.95 else ('MODERATE' if centroid_cosine < 0.98 else 'LOW')
    }
    
    output_path = 'results/embedding_drift.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")
    print("="*60)

if __name__ == '__main__':
    main()
