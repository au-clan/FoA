import ast
from collections import defaultdict

# Step 1: Read log
with open('new_test_results.log', 'r') as f:
    lines = f.readlines()

# Step 2: Parse lines
data = []
for line in lines:
    try:
        # Only take the part after the "-"
        dict_part = line.split(" - ", 1)[1]
        entry = ast.literal_eval(dict_part)
        data.append(entry)
    except Exception as e:
        print(f"Skipping line due to error: {e}")

# Step 3: Grouping
grouped = defaultdict(list)
for entry in data:
    key = (entry['num_reflexions'], entry['reflexion_type'])
    grouped[key].append(entry)

# Step 4: Summarize
summary = {}
for key, entries in grouped.items():
    avg_score = sum(e['score'] for e in entries) / len(entries)
    avg_token_cost = sum(e['token_cost'] for e in entries) / len(entries)
    summary[key] = {
        'avg_score': avg_score,
        'avg_token_cost': avg_token_cost,
        'count': len(entries)
    }

# Step 5: Print nicely
for key, stats in summary.items():
    num_reflexions, reflexion_type = key
    print(f"Reflexions: {num_reflexions}, Type: {reflexion_type}")
    print(f"  Avg Score: {stats['avg_score']:.2f}")
    print(f"  Avg Token Cost: {stats['avg_token_cost']:.1f}")
    print(f"  Num Samples: {stats['count']}")
    print()
