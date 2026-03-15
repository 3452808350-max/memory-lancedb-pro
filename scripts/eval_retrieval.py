#!/usr/bin/env python3
"""
Full Retrieval Benchmark

完整的检索性能评估。

Usage:
    python scripts/eval_retrieval.py --dataset memory_db/
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import lancedb
from tqdm import tqdm

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser(description='Full Retrieval Benchmark')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='BAAI/bge-m3', help='Embedding model')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K retrieval')
    args = parser.parse_args()
    
    print(f"📊 Loading dataset from {args.dataset}...")
    memories = load_jsonl(f"{args.dataset}/memories.jsonl")
    queries = load_jsonl(f"{args.dataset}/queries.jsonl")
    print(f"✅ Loaded {len(memories)} memories, {len(queries)} queries")
    
    # Load model
    print(f"\n🤖 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Create LanceDB table
    print("\n🗄️  Creating LanceDB table...")
    db = lancedb.connect('/tmp/lancedb_benchmark')
    
    # Generate memory embeddings
    print("\n🚀 Generating memory embeddings...")
    memory_texts = [m.get('text', '') for m in memories]
    memory_embeddings = model.encode(memory_texts, batch_size=64, show_progress_bar=True)
    
    # Create table
    table_data = [{
        'id': m['id'],
        'text': m['text'],
        'embedding': emb.tolist()
    } for m, emb in zip(memories, memory_embeddings)]
    
    try:
        db.drop_table('memories')
    except:
        pass
    
    table = db.create_table('memories', table_data)
    
    # Evaluate retrieval
    print("\n🔍 Evaluating retrieval performance...")
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    mrr_sum = 0
    
    for q in tqdm(queries):
        target_id = q.get('target_memory_id')
        query_text = q.get('query')
        
        # Vector search
        query_emb = model.encode([query_text])[0]
        results = table.search(query_emb.tolist()).limit(args.top_k).to_list()
        
        retrieved_ids = [r['id'] for r in results]
        
        # Metrics
        if target_id in retrieved_ids[:1]:
            recall_at_1 += 1
        if target_id in retrieved_ids[:5]:
            recall_at_5 += 1
        if target_id in retrieved_ids:
            recall_at_10 += 1
        
        try:
            rank = retrieved_ids.index(target_id) + 1
            mrr_sum += 1 / rank
        except ValueError:
            pass
    
    n = len(queries)
    r1 = recall_at_1 / n
    r5 = recall_at_5 / n
    r10 = recall_at_10 / n
    mrr = mrr_sum / n
    
    # Report
    print("\n" + "="*60)
    print("📈 Retrieval Benchmark Results")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Memories: {len(memories)}, Queries: {n}")
    print(f"Model: {args.model}")
    print(f"\n📊 Metrics:")
    print(f"  Recall@1:  {r1:.3f}")
    print(f"  Recall@5:  {r5:.3f}")
    print(f"  Recall@10: {r10:.3f}")
    print(f"  MRR:       {mrr:.3f}")
    print("="*60)
    
    # Save results
    results = {
        'dataset': args.dataset,
        'num_memories': len(memories),
        'num_queries': n,
        'model': args.model,
        'recall_at_1': r1,
        'recall_at_5': r5,
        'recall_at_10': r10,
        'mrr': mrr
    }
    
    output_path = 'results/benchmark.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")

if __name__ == '__main__':
    main()
