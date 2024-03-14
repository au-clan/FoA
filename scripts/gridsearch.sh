#!/bin/bash

# Define arrays for each parameter
k__values=(1 2 3)
max_steps__values=(5 7 9)
n_agents__values=(3 5 10)
backtrack__values=(0 0.25 0.5 0.75 0.9)
total=0
# Iterate over all combinations of parameters
for k in "${k__values[@]}"; do
    for max_steps in "${max_steps__values[@]}"; do
        for n_agents in "${n_agents__values[@]}"; do
            for backtrack in "${backtrack__values[@]}"; do
                echo "Running with: k=$k, max_steps=$max_steps, N n_agents=$n_agents, backtrack=$backtrack"
                 python async_implementation/experiments/gameof24.py --k $k --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set validation
                ((total++))
            done
        done
    done
done

echo "Total: $total"
