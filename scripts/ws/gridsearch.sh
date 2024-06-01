#!/bin/bash

# Define arrays for each parameter
# # num_steps__values=(5 10 15)
# # num_agents__values=(3 5)
# # backtrack__values=(0 0.3 0.6)
# # k__values=(1 3 5)
# # prompting__values=(react act)

num_steps__values=(20)
num_agents__values=(20 25)
backtrack__values=(0)
k__values=(5 3)
prompting__values=(react)

iteration=0


n_iterations=$((${#num_steps__values[@]} * ${#num_agents__values[@]} * ${#backtrack__values[@]} * ${#k__values[@]} * ${#prompting__values[@]}))
echo "n_iterations: $n_iterations"

# Iterate over all combinations of parameters
for num_steps in "${num_steps__values[@]}"; do
    for num_agents in "${num_agents__values[@]}"; do
        for backtrack in "${backtrack__values[@]}"; do
            for k in "${k__values[@]}"; do
                for prompting in "${prompting__values[@]}"; do
                    echo "Progress: $((iteration+1))/$n_iterations"
                    python async_implementation/experiments/ws.py --set test1 --num_agents $num_agents --backtrack $backtrack --num_steps $num_steps --k $k --prompting $prompting --send_email 
                    ((iteration++))
                done
            done
        done
    done
done


echo "Total: $iteration"