#!/bin/bash

# Define arrays for each parameter
max_steps__values=(9 8 7 6 5)
n_agents__values=(15 10 7)
backtrack__values=(0.1 0.2 0.5)
iteration=0
# Iterate over all combinations of parameters

n_iterations=$((${#max_steps__values[@]} * ${#n_agents__values[@]} * ${#backtrack__values[@]}))
echo "n_iterations: $n_iterations"

for max_steps in "${max_steps__values[@]}"; do
    for n_agents in "${n_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            python async_implementation/experiments/gameof24_cached.py --k 1 --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set test
            echo "Progress: $iteration/$n_iterations"
            ((iteration++))
        done
    done
done


echo "Total: $iteration"