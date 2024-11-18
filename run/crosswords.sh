# #!/bin/bash

# # Define arrays for each parameter
# max_steps__values=(20 30)
# n_agents__values=(2 3)
# backtrack__values=(0.5 0.75 0.9)
# k__values=(3)
# total=0

# for max_steps in "${max_steps__values[@]}"; do
#     for n_agents in "${n_agents__values[@]}"; do
#         for backtrack in "${backtrack__values[@]}"; do
#             for k in "${k__values[@]}"; do
#                 python run/crosswords.py --set validation --n_agents $n_agents --max_steps $max_steps --k $k --backtrack $backtrack  --send_email
#                 ((total++))
#             done
#         done
#     done
# done

# echo "Total: $total"

python run/crosswords.py --set test --n_agents 3 --max_steps 30 --k 3 --backtrack 0.5  --send_email