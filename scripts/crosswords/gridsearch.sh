#!/bin/bash

# Define arrays for each parameter
max_steps__values=(8 6 4)
n_agents__values=(2)
backtrack__values=(0 0.2 0.5)
k__values=(1 2)
iteration=0
n_iterations=$((${#max_steps__values[@]} * ${#n_agents__values[@]} * ${#backtrack__values[@]} * ${#k__values[@]}))
echo "n_iterations: $n_iterations"

for max_steps in "${max_steps__values[@]}"; do
    for n_agents in "${n_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            for k in "${k__values[@]}"; do
                echo "Progress: $iteration/$n_iterations"
                python async_implementation/experiments/crosswords_cached.py --k $k --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set test --send_email
                ((iteration++))
            done
        done
    done
done

echo "Total: $iteration"

max_steps__values=(6 4)
n_agents__values=(6 4)
backtrack__values=(0 0.2 0.5)
k__values=(1 2)
iteration=0
n_iterations=$((${#max_steps__values[@]} * ${#n_agents__values[@]} * ${#backtrack__values[@]} * ${#k__values[@]}))
echo "n_iterations: $n_iterations"

for max_steps in "${max_steps__values[@]}"; do
    for n_agents in "${n_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            for k in "${k__values[@]}"; do
                echo "Progress: $iteration/$n_iterations"
                python async_implementation/experiments/crosswords_cached.py --k $k --max_steps $max_steps --n_agents $n_agents --backtrack $backtrack --set test --send_email
                ((iteration++))
            done
        done
    done
done

echo "Total: $iteration"