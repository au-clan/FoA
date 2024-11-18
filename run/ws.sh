# #!/bin/bash

# max_steps__values=(10 15)
# n_agents__values=(20 30 40)
# backtrack__values=(0.5 0.95)
# n_evaluations__values=(10 20)
# origin_value__values=(1)
# total=0

# for max_steps in "${max_steps__values[@]}"; do
#     for n_agents in "${n_agents__values[@]}"; do
#         for backtrack in "${backtrack__values[@]}"; do
#             for n_evaluations in "${n_evaluations__values[@]}"; do
#                 for origin_value in "${origin_value__values[@]}"; do
#                     python run/ws.py --set test --num_agents $n_agents --num_steps $max_steps --k 4 --n_evaluations $n_evaluations --backtrack $backtrack --resampling linear --origin_value $origin_value --send_email
#                     ((total++))
#                 done
#             done
#         done
#     done
# done
# echo "Total: $total"


# FOA
#python run/ws.py --set test --num_agents 15 --num_steps 10 --k 4 --backtrack 0.5 --resampling linear --n_evaluations 10 --send_email
#python run/ws.py --set test --num_agents 30 --num_steps 10 --k 3 --backtrack 0.95 --resampling linear --n_evaluations 20 --origin_value 1 --send_email
#python run/ws.py --set test --num_agents 30 --num_steps 10 --k 4 --backtrack 0.95 --resampling linear --n_evaluations 20 --origin_value 1 --send_email
#python run/ws.py --set test --num_agents 40 --num_steps 10 --k 4 --backtrack 0.95 --resampling linear --n_evaluations 10 --origin_value 1 --send_email

# REACT
python run/ws_react.py --set test --num_agents 1 --num_steps 15 --k 0 --backtrack 0 --resampling linear --n_evaluations 0 --send_email --repeats 90