#!/bin/bash

# Define arrays for each parameter
seed__values=(4)
# Iterate over all combinations of parameters

for seed in "${seed__values[@]}"; do
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 7 --backtrack 0.5 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 6 --n_agents 7 --backtrack 0.1 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 10 --backtrack 0.1 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 4 --n_agents 10 --backtrack 0.5 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 4 --n_agents 10 --backtrack 0 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 7 --n_agents 15 --backtrack 0.1 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 15 --backtrack 0.2 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 4 --n_agents 15 --backtrack 0 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 5 --backtrack 0.25 --set test --seed $seed
    #python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 5 --backtrack 0.5 --set test --seed $seed
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 10 --backtrack 0.25 --set test --seed $seed
done

seed__values=(1 2)
# Iterate over all combinations of parameters

for seed in "${seed__values[@]}"; do
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 5 --backtrack 0.25 --set test --seed $seed
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 9 --n_agents 5 --backtrack 0.5 --set test --seed $seed
    python async_implementation/experiments/gameof24.py --k 1 --max_steps 5 --n_agents 10 --backtrack 0.25 --set test --seed $seed
done