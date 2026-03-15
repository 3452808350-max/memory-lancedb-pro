#!/usr/bin/env python3
"""
Prepare Standard Dataset

将 synthetic_perltqa 数据转换为标准格式。

Usage:
    python scripts/prepare_dataset.py --input ../synthetic_perltqa --output memory_db/
"""

import json
import argparse
import os
from pathlib import Path

def convert_scale(input_dir, output_dir, scale):
    """转换单个规模的数据"""
    memories_in = f"{input_dir}/memories_{scale}.json"
    queries_in = f"{input_dir}/queries_{scale}.json"
    
    if not os.path.exists(memories_in) or not os.path.exists(queries_in):
        print(f"⚠️  Skipping {scale} (files not found)")
        return
    
    # Convert memories to JSONL
    with open(memories_in, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    with open(f"{output_dir}/memories_{scale}.jsonl", 'w', encoding='utf-8') as f:
        for m in memories:
            # Standard format
            std_m = {
                'id': m.get('id'),
                'text': m.get('content'),
                'type': m.get('type'),
                'timestamp': m.get('timestamp', ''),
                'metadata': m.get('metadata', {})
            }
            f.write(json.dumps(std_m, ensure_ascii=False) + '\n')
    
    # Convert queries to JSONL
    with open(queries_in, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    
    with open(f"{output_dir}/queries_{scale}.jsonl", 'w', encoding='utf-8') as f:
        for q in queries:
            # Standard format with ground truth
            std_q = {
                'id': q.get('id'),
                'query': q.get('query'),
                'type': q.get('type', 'unknown'),
                'target_memory_id': q.get('target_memory_id'),
                'target_memory_content': q.get('target_memory_content'),
                'expected_keywords': q.get('expected_keywords', [])
            }
            f.write(json.dumps(std_q, ensure_ascii=False) + '\n')
    
    print(f"✅ Converted {scale}: {len(memories)} memories, {len(queries)} queries")

def main():
    parser = argparse.ArgumentParser(description='Prepare Standard Dataset')
    parser.add_argument('--input', default='../synthetic_perltqa', help='Input directory')
    parser.add_argument('--output', default='memory_db/', help='Output directory')
    args = parser.parse_args()
    
    print(f"📂 Converting data from {args.input} to {args.output}")
    
    os.makedirs(args.output, exist_ok=True)
    
    scales = ['baseline', 'small', 'medium', 'medium-large', 'large']
    
    for scale in scales:
        convert_scale(args.input, args.output, scale)
    
    # Create stats summary
    stats = {}
    for scale in scales:
        mem_file = f"{args.output}/memories_{scale}.jsonl"
        qry_file = f"{args.output}/queries_{scale}.jsonl"
        
        if os.path.exists(mem_file):
            with open(mem_file) as f:
                mem_count = sum(1 for _ in f)
            with open(qry_file) as f:
                qry_count = sum(1 for _ in f)
            
            stats[scale] = {
                'memories': mem_count,
                'queries': qry_count
            }
    
    with open(f"{args.output}/stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📊 Dataset Statistics:")
    for scale, s in stats.items():
        print(f"  {scale}: {s['memories']} memories, {s['queries']} queries")
    
    print(f"\n💾 Stats saved to {args.output}/stats.json")

if __name__ == '__main__':
    main()
