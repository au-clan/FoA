import ast
from collections import defaultdict

#Reading
with open('new_test_results.log', 'r') as f:
    lines = f.readlines()

#Parsing
data = []
for line in lines:
    try:
        # Only take the part after the "-"
        dict_part = line.split(" - ", 1)[1]
        entry = ast.literal_eval(dict_part)
        data.append(entry)
    except Exception as e:
        print(f"Skipping line due to error: {e}")

# Define mapping
easy_puzzles = {"1 1 4 6", "1 1 11 11", "6 6 6 6", "1 1 1 12", "1 1 2 12"}
medium_puzzles = {"2 4 7 7", "3 6 6 10", "4 7 9 11", "2 2 3 5", "2 5 7 9"}
hard_puzzles = {"2 4 10 10", "5 5 7 11", "1 3 4 6", "5 7 7 11", "3 3 7 13"}

# Assign difficulty based on puzzle
for entry in data:
    puzzle = entry['puzzle']
    if puzzle in easy_puzzles:
        entry['difficulty'] = 'easy'
    elif puzzle in medium_puzzles:
        entry['difficulty'] = 'medium'
    elif puzzle in hard_puzzles:
        entry['difficulty'] = 'hard'
    else:
        entry['difficulty'] = 'unknown'  # In case of unexpected puzzle

#grouping
grouped = defaultdict(list)
for entry in data:
    key = (entry['difficulty'], entry['num_reflexions'], entry['reflexion_type'])
    grouped[key].append(entry)

# Summarize
summary = {}
for key, entries in grouped.items():
    avg_score = sum(e['score'] for e in entries) / len(entries)
    avg_token_cost = sum(e['token_cost'] for e in entries) / len(entries)
    summary[key] = {
        'avg_score': avg_score,
        'avg_token_cost': avg_token_cost,
        'count': len(entries)
    }

# Print nicely
for key, stats in summary.items():
    difficulty, num_reflexions, reflexion_type = key
    print(f"Difficulty: {difficulty}, Reflexions: {num_reflexions}, Type: {reflexion_type}")
    print(f"  Avg Score: {stats['avg_score']:.2f}")
    print(f"  Avg Token Cost: {stats['avg_token_cost']:.1f}")
    print(f"  Num Samples: {stats['count']}")
    print()
