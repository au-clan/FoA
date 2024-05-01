#!/bin/bash

# Define arrays for each parameter
seed__values=(0)
# Iterate over all combinations of parameters

for seed in "${seed__values[@]}"; do
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 4 --n_agents 2 --backtrack 0 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 6 --n_agents 2 --backtrack 0 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 2 --max_steps 6 --n_agents 2 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 2 --max_steps 6 --n_agents 2 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 6 --n_agents 4 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 8 --n_agents 2 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 4 --n_agents 6 --backtrack 0.5 --set test --seed $seed --send_email
    python async_implementation/experiments/crosswords_cached.py --k 3 --max_steps 12 --n_agents 2 --backtrack 0.5 --set test --seed $seed --send_email

done