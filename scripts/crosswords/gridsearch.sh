#!/bin/bash

# Define arrays for each parameter
max_steps__values=(10 8 6 4)
n_agents__values=(6 4 2)
backtrack__values=(0 0.2 0.5)
k__values=(1 2 3)
total=0

for max_steps in "${max_steps__values[@]}"; do
    for n_agents in "${n_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            for k in "${k__values[@]}"; do
                python async_implementation/experiments/crosswords.py --k $k --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set validation --send_email
                ((total++))
            done
        done
    done
done

echo "Total: $total"