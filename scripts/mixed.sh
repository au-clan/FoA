#!/bin/bash

# Define arrays for each parameter
seed__values=(0)
# Iterate over all combinations of parameters

for seed in "${seed__values[@]}"; do
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 4 --n_agents 10 --backtrack 0 --set test --seed $seed --send_email
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 7 --backtrack 0.1 --set test --seed $seed --send_email
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 15 --backtrack 0.2 --set test --seed $seed --send_email
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 7 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords.py --k 3 --max_steps 6 --n_agents 2 --backtrack 0 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords.py --k 3 --max_steps 4 --n_agents 6 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords.py --k 3 --max_steps 6 --n_agents 4 --backtrack 0.5 --set test --seed $seed --send_email
done