# test_sets=(test1 test2 test3)

# for test_set in "${test_sets[@]}"; do
#     python async_implementation/experiments/ws.py --set $test_set --num_agents 5 --backtrack 0.6 --num_steps 15 --k 4 --prompting act --repeats 1 --send_email
# done

# python async_implementation/experiments/ws.py --set test --num_agents 5 --backtrack 0.6 --num_steps 15 --k 4 --prompting act --repeats 1 --send_email
python async_implementation/experiments/ws.py --set test --num_agents 15 --backtrack 0.1 --num_steps 11 --k 5 --prompting react --repeats 1 --send_email