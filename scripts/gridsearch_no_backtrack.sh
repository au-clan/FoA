#!/bin/bash

# Define arrays for each parameter
max_steps__values=(7)
n_agents__values=(7)
backtrack__values=(0)
total=0
# Iterate over all combinations of parameters

for max_steps in "${max_steps__values[@]}"; do
    for n_agents in "${n_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            echo "Running with: k=1, max_steps=$max_steps, N n_agents=$n_agents, backtrack=$backtrack"
                python async_implementation/experiments/gameof24.py --k 1 --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set test
            ((total++))
        done
    done
done


echo "Total: $total"
