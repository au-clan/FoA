# I'm keeping the same comments, etc everywhere, so that later it's easier to merge experiments.gameof24.py and experiments.crosswords.py (hydra?)

import asyncio
# set up logging
import logging
import os
import json
import random
import argparse

import numpy as np
import pandas as pd
from diskcache import Cache
from datetime import datetime
from collections import Counter


# TODO: Not sure if this is correct, I didn't know how else to handle the package paths
import sys
sys.path.append(os.getcwd()) # Project root!!

from async_engine.cached_api import CachedOpenAIAPI
from async_engine.round_robin_manager import AsyncRoundRobin
from async_engine.batched_api import BatchingAPI
from async_implementation.agents.crosswords import CrosswordsAgent
from async_implementation.states.crosswords import CrosswordsState
from async_implementation.resampling.resampler import Resampler
from utils import create_folder, email_notification

logger = logging.getLogger("experiments")
logger.setLevel(logging.DEBUG) # Order : debug < info < warning < error < critical
log_folder = f"logs/{datetime.now().date()}/crosswords/{datetime.now().strftime('%H')}:00/" # Folder in which logs will be saved (organized daily)
create_folder(log_folder)

# you should use the same cache for every instance of CachedOpenAIAPI
# that way we never pay for the same request twice
assert os.path.exists(
    "./caches/"), "Please run the script from the root directory of the project. To make sure all caches are created correctly."
cache = Cache("./caches/async_api_cache", size_limit=int(2e10))

# get OPENAI_API_KEY from env
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "Please set the OPENAI_API_KEY environment variable"

api_config = {
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 1,
    "request_timeout": 45,
    "model": "gpt-3.5-turbo"
}

api = CachedOpenAIAPI(cache, api_config)
limiter = AsyncRoundRobin()
# ToDo, this is a bit hacky. OpenAI allows multiple parallel requests per key, so we add the same key multiple times
N = 4
for _ in range(N):
    limiter.add_resource(data=OPENAI_API_KEY)

# Set up Crosswords puzzles
path = "data/datasets/mini0505.json"
with open(path, "r") as file:
    dataset = json.load(file)

# ToDo: this should probably be moved to its own file
# for now I'm keeping it here, for easier debugging
#async def foa_gameof24(puzzle_idx: int, num_agents=3, k=2, backtrack=0.8):
async def foa_crosswords(puzzle_idx, foa_options):
    randomness = puzzle_idx
    random.seed(randomness)

    resampler = Resampler(randomness)

    data, board_gt = dataset[puzzle_idx] # Data is the list of clues, board_gt is the ground truth  board
    ans_gt = CrosswordsState.get_ans(board_gt) # Get the ground truth answers

    # Set up logging
    log = {puzzle_idx: {f"Agent {i}":{} for i in range(foa_options["num_agents"])}}
    for a in range(foa_options["num_agents"]):
        log[puzzle_idx][f"Agent {a}"]["action"] = []
        log[puzzle_idx][f"Agent {a}"]["value"] = []
        log[puzzle_idx][f"Agent {a}"]["resample"] = []
        log[puzzle_idx][f"Agent {a}"]["metrics"] = []

    # Records
    r = {"idx":["INIT"], "values":[foa_options["origin_value"]], "states":[CrosswordsState(data=data, board_gt=board_gt, ans_gt=ans_gt, steps=[], randomness=random.randint(0, 1000))]}

    # Set up states
    states = []
    for _ in range(foa_options["num_agents"]):
        states.append(CrosswordsState(data=data, board_gt=board_gt, ans_gt=ans_gt, steps=[], randomness=random.randint(0, 1000)))

    num_steps = foa_options["num_steps"]
    for step in range(num_steps):
        print(f"Step {step}")

        # Step : make one step for each state
        agent_coroutines = []
        for state in states:
            agent_coroutines.append(CrosswordsAgent.step(state, api))
        states, actions = zip(* await asyncio.gather(*agent_coroutines))

        for i, (state, action) in enumerate(zip(states, actions)):
            log[puzzle_idx][f"Agent {i}"]["action"].append(". ".join(action))


        # TODO: Verification 
        verifications = ...
        
        # NOTE: Pruning can be employed as soon as verification is implemented
        #states = [state for state, verification in zip(states, verifications) if verification == {"r":0}]
        #new_states, _ = resampler.resample(r, foa_options["num_agents"]-len(states), foa_options["resampling_method"])
        #states += new_states

        # Depreciation
        r["values"] = [value * foa_options["backtrack"] if i != 0 else value * (foa_options["backtrack"] ** 2) for i, value in enumerate(r["values"])]

        # Resampling : every k steps, evaluate and resample
        #if step < num_steps - 1 and step % foa_options["k"] == 0: # <- Correct one : changing it just fo debugging
        if step < num_steps and step % foa_options["k"] == 1:

            # Evaluation : each of the current states is given a value
            value_coroutines = []
            for state in states:
                value_coroutines.append(CrosswordsAgent.evaluate(state, api))
            values = await asyncio.gather(*value_coroutines)

            # Update records
            for i, (state, value) in enumerate(zip(states, values)):
                r["idx"].append(f"{step}.{i}")
                r["values"].append(value)
                r["states"].append(state)
            
            # Logging : Add NAs for steps that were not evaluated
            while len(log[puzzle_idx]["Agent 0"]["value"]) < step:
                for a in range(foa_options["num_agents"]):
                    log[puzzle_idx][f"Agent {a}"]["value"].append("NA")
                    log[puzzle_idx][f"Agent {a}"]["resample"].append("NA")
            
            # Logging : Evaluation
            for i, v in enumerate(values):
                log[puzzle_idx][f"Agent {i}"]["value"].append(v)
                
            # Resampling
            states, resampled_indices = resampler.resample(r, foa_options["num_agents"], foa_options["resampling_method"])
            
            # Logging : Resampling
            for i, j in enumerate(resampled_indices):
                log[puzzle_idx][f"Agent {i}"]["resample"].append(f"{r['idx'][j]}")
            
        # Logging : metrics
        for i, state in enumerate(states):
            log[puzzle_idx][f"Agent {i}"]["metrics"].append(state.get_metrics())
                
    #TODO: Final logging
            
    return states, log


async def run(run_options:dict, foa_options:dict):
    """
    Inputs
        difficulty: Selects the starting index
        sample_size: Selects the number of experiments to run
    """
    game_coroutines = []
    log = {}

    # Run FoA for each puzzle
    for idx in range(run_options["difficulty"]*10, run_options["difficulty"]*10+run_options["sample_size"]):
        game_coroutines.append(foa_crosswords(idx, foa_options))
    results = await asyncio.gather(*game_coroutines)
    
    # Merge logs for each run
    logs = [log for (game, log) in results]
    for l in logs:
        log.update(l)
    
    # Save merged logs
    with open(log_folder + log_file, 'w+') as f:
        json.dump(log, f, indent=4)
    
    # Return game states for each run
    game_states = [game for (game, log) in results]
    return game_states





#################
### Execution ###
#################

def parse_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--difficulty", type=int, choices=list(range(10)), default=0)
    args.add_argument("--n_samples", type=int, default=1)
    args.add_argument("--n_agents", type=int, default=2)
    args.add_argument("--back_coef", type=float, default=0.8)
    args.add_argument("--max_steps", type=int, default=2)
    args.add_argument("--resampling", type=str, choices=["linear", "logistic", "max", "percentile"], default="linear")
    args = args.parse_args()
    return args
args = parse_args()

# Select parameters
difficulty = args.difficulty               # Starting idx = 10 * difficulty
sample_size = args.n_samples               # Ending idx   = 10 * difficulty + sample_size
num_agents = args.n_agents                 # Number of agents
k = 2                                      # Resampling every <k> steps
origin_value = 0                           # The evaluation of the origin state #TODO: Change 3 to num_evaluations
num_steps = args.max_steps                 # Total number of steps FoA executes
backtrack = args.back_coef                 # Backtrack decaying coefficient
resampling_method = args.resampling        # Resampling method



# Just for now so it's easier to change values and reduce noise
log_file = f"{difficulty}diff_{num_agents}agents_{num_steps}steps_{sample_size}sample_{k}k_{origin_value}origin_{backtrack}backtrack_{resampling_method}-resampling.json"

run_options = {
    "difficulty":difficulty,
    "sample_size":sample_size}

foa_options = {
    "num_agents":num_agents,
    "k":k,
    "origin_value":origin_value,
    "num_steps":num_steps,
    "backtrack":backtrack,
    "resampling_method":resampling_method
}


# Use batching API
api = BatchingAPI(api, limiter, batch_size=foa_options["num_agents"], timeout=10)

# Run
results = asyncio.run(run(run_options, foa_options))

# TODO: Verification/accuracy
accuracy = 0

# Send email notification
send_email = False
if send_email:
    subject = log_file
    message = f"Accuracy : {accuracy}"
    email_notification(subject=subject, message=message)

