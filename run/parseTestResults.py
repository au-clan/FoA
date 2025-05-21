import ast 
from collections import defaultdict
import sys
import os
sys.path.append(os.getcwd())
sys.stdout.reconfigure(encoding='utf-8')

# Reading
with open('reflexionLogs/stepwise_LLM.log', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Parsing
data = []
for line in lines:
    try:
        # Only take the part after the "-"
        dict_part = line.split(" - ", 1)[1]
        entry = ast.literal_eval(dict_part)
        data.append(entry)
    except Exception as e:
        print(f"Skipping line due to error: {e}")

# Grouping
grouped = defaultdict(list)
for entry in data:
    key = (entry['num_reflexions'], entry['reflexion_type'])
    grouped[key].append(entry)

# Summarize
summary = {}
for key, entries in grouped.items():
    avg_score = sum(e['score'] for e in entries) / len(entries)
    avg_tokens_used = sum(e['tokens_used'] for e in entries) / len(entries)
    avg_total_tokens = sum(e['total_tokens'] for e in entries) / len(entries)
    avg_price_used = sum(e['price_used'] for e in entries) / len(entries)
    avg_price_total = sum(e['price_total'] for e in entries) / len(entries)

    summary[key] = {
        'avg_score': avg_score,
        'avg_tokens_used': avg_tokens_used,
        'avg_total_tokens': avg_total_tokens,
        'avg_price_used': avg_price_used,
        'avg_price_total': avg_price_total,
        'count': len(entries)
    }

# Print nicely
for key, stats in summary.items():
    num_reflexions, reflexion_type = key
    print(f"Reflexions: {num_reflexions}, Type: {reflexion_type}")
    print(f"  Avg Score:       {stats['avg_score']:.2f}")
    print(f"  Avg Tokens Used: {stats['avg_tokens_used']:.1f}")
    print(f"  Avg Total Tokens:{stats['avg_total_tokens']:.1f}")
    print(f"  Avg Cost Used:   ${stats['avg_price_used']:.6f}")
    print(f"  Avg Total Cost:  ${stats['avg_price_total']:.6f}")
    print(f"  Num Samples:     {stats['count']}")
    print()
