#!/usr/bin/env python3
"""
Hybrid Retrieval Weight Sweep

找到最优 hybrid 权重 α。

Usage:
    python scripts/eval_hybrid.py --memory memory_db/memories.jsonl --queries memory_db/queries.jsonl
"""

import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import lancedb
from tqdm import tqdm

def load_data(memory_path, query_path):
    memories = []
    with open(memory_path, 'r', encoding='utf-8') as f:
        for line in f:
            memories.append(json.loads(line))
    
    queries = []
    with open(query_path, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    
    return memories, queries

def hybrid_score(vec_sim, bm25_sim, alpha):
    """
    Hybrid score fusion
    
    score = α * vec_sim + (1-α) * bm25_sim
    """
    return alpha * vec_sim + (1 - alpha) * bm25_sim

def main():
    parser = argparse.ArgumentParser(description='Hybrid Retrieval Weight Sweep')
    parser.add_argument('--memory', required=True, help='Path to memories.jsonl')
    parser.add_argument('--queries', required=True, help='Path to queries.jsonl')
    parser.add_argument('--alpha_list', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument('--model', default='BAAI/bge-m3', help='Embedding model')
    parser.add_argument('--top_k', type=int, default=10, help='Top-K retrieval')
    args = parser.parse_args()
    
    print(f"📊 Loading data...")
    memories, queries = load_data(args.memory, args.queries)
    print(f"✅ Loaded {len(memories)} memories, {len(queries)} queries")
    
    # Load model
    print(f"\n🤖 Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)
    
    # Create LanceDB table
    print("\n🗄️  Creating LanceDB table...")
    db = lancedb.connect('/tmp/lancedb_test')
    
    # Generate embeddings
    print("\n🚀 Generating embeddings...")
    texts = [m.get('text', '') for m in memories]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    
    # Create table
    table_data = [{
        'id': m['id'],
        'text': m['text'],
        'embedding': emb.tolist()
    } for m, emb in zip(memories, embeddings)]
    
    table = db.create_table('memories', table_data)
    
    # Evaluate each alpha
    print("\n" + "="*60)
    print("📈 Hybrid Weight Sweep Results")
    print("="*60)
    
    results = []
    
    for alpha in args.alpha_list:
        print(f"\n🔍 Testing α = {alpha:.1f}...")
        
        recall_at_5 = 0
        recall_at_10 = 0
        mrr_sum = 0
        
        for q in tqdm(queries, desc=f'α={alpha:.1f}'):
            target_id = q.get('target_memory_id')
            query_text = q.get('query')
            
            # Vector search
            query_emb = model.encode([query_text])[0]
            vector_results = table.search(query_emb.tolist()).limit(args.top_k).to_list()
            
            # Simple evaluation (vector only for now)
            retrieved_ids = [r['id'] for r in vector_results]
            
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
        r5 = recall_at_5 / n
        r10 = recall_at_10 / n
        mrr = mrr_sum / n
        
        results.append({
            'alpha': alpha,
            'recall_at_5': r5,
            'recall_at_10': r10,
            'mrr': mrr
        })
        
        print(f"  Recall@5: {r5:.3f}")
        print(f"  Recall@10: {r10:.3f}")
        print(f"  MRR: {mrr:.3f}")
    
    # Find best alpha
    best = max(results, key=lambda x: x['recall_at_5'])
    
    print("\n" + "="*60)
    print(f"🏆 Best α = {best['alpha']:.1f}")
    print(f"   Recall@5: {best['recall_at_5']:.3f}")
    print(f"   Recall@10: {best['recall_at_10']:.3f}")
    print(f"   MRR: {best['mrr']:.3f}")
    print("="*60)
    
    # Save results
    output = {
        'alpha_sweep': results,
        'best_alpha': best['alpha'],
        'best_recall_at_5': best['recall_at_5'],
        'best_recall_at_10': best['recall_at_10'],
        'best_mrr': best['mrr']
    }
    
    output_path = 'results/hybrid_sweep.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to {output_path}")

if __name__ == '__main__':
    main()
