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

sys.path.append(os.getcwd())  # Project root!!

from async_engine.cached_api_namespace import CachedOpenAIAPI
from async_engine.round_robin_manager import AsyncRoundRobin
from async_engine.batched_api import BatchingAPI
from async_implementation.agents.gameof24 import GameOf24Agent
from async_implementation.states.gameof24 import GameOf24State
from async_implementation.resampling.resampler import Resampler
from data.data import GameOf24Data
from utils import create_folder, email_notification, compare_json_files

logger = logging.getLogger("experiments")

logger.setLevel(logging.DEBUG) # Order : debug < info < warning < error < critical
log_folder = f"logs/{datetime.now().date()}/gameof24/{datetime.now().strftime('%H')}:00/" # Folder in which logs will be saved (organized daily)
create_folder(log_folder)

# you should use the same cache for every instance of CachedOpenAIAPI
# that way we never pay for the same request twice
assert os.path.exists(
    "./caches/"), "Please run the script from the root directory of the project. To make sure all caches are created correctly."
cache = Cache("./caches/test_cache", size_limit=int(2e10))

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

# Setting up the data
data = GameOf24Data()


# ToDo: this should probably be moved to its own file
# for now I'm keeping it here, for easier debugging
# async def foa_gameof24(puzzle_idx: int, num_agents=3, k=2, backtrack=0.8):
async def foa_gameof24(api, limiter, puzzle_idx, puzzle, foa_options, barrier):
    # Use batching API
    step_api = BatchingAPI(api, limiter, batch_size=foa_options["num_agents"], timeout=10)
    eval_api = BatchingAPI(api, limiter, batch_size=foa_options["num_agents"]*3, timeout=10)

    randomness = puzzle_idx
    random.seed(randomness)

    resampler = Resampler(randomness)
    
    log = {puzzle_idx:{"puzzle": puzzle}}


    # State identifier shows the step and the agent where the state was visited eg. 0.1 means step 0, agent 1
    state_records = [] # List of states [(state_identifier, state_value, state)]
    state_records.append(("INIT", foa_options["origin_value"], GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000))))

    # set up states
    states = []

    for _ in range(foa_options["num_agents"]):
        states.append(GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000)))

    solution_found = False ### DEBUG: Just for now until I figure out something better
    num_steps = foa_options["num_steps"]
    for step in range(num_steps):

        ### DEBUG: Just for now until I figure out something better
        if solution_found:
            await barrier.wait()
            continue
        
        #print(f"Step {step}")
        log[puzzle_idx][f"Step {step}"] = {}

        # Step : make one step for each state
        agent_coroutines = []
        for agent_id, state in enumerate(states):
            agent_coroutines.append(GameOf24Agent.step(state, step_api, namespace=(puzzle_idx, f"Agent: {agent_id}", f"Step : {step}")))
        states = await asyncio.gather(*agent_coroutines)

        # Log steps
        log[puzzle_idx][f"Step {step}"]["steps"] = [" || ".join(state.steps) for state in states]
        assert len(step_api.futures) == 0, f"API futures should be empty, but are {len(step_api.futures)}"

        # Depreciation : old state values are decayed by the backtrack coefficient
        state_records = [(idx, value*foa_options["backtrack"], state) for idx, value, state in state_records]

        # Verification : After each step we verify if the answer is found and if so we break    
        verifications = [GameOf24Agent.verify(state) for state in states]   # {"r":1} Finished correctly
        if {"r":1} in verifications:                                        # {"r":-1} Finished incorrectly / "Ivalid state"
            ### DEBUG: Just for now until I figure out something better - Normally you want to break here
            solution_found = True
            await barrier.wait()
            continue                                                        # {"r":0} Not finished                

        # Pruning # TODO : Pruned states should NOT be re-evaluated. 
        invalid_state_indices = [i for i, verification in enumerate(verifications) if verification["r"] == -1]
        new_states, pruned_indices = resampler.resample(state_records, len(invalid_state_indices), foa_options["resampling_method"], include_init=False)
        states = [new_states.pop(0) if i in invalid_state_indices else state for i, state in enumerate(states)]

        # Log pruning
        log[puzzle_idx][f"Step {step}"]["Pruning"] = [f"{i}<-{state_records[j][0]}" for i, j in zip(invalid_state_indices, pruned_indices)]

        # Synchronize experiments
        ### DEBUG: Normally syncrhonisation is done before resampling not after every step
        await barrier.wait()

        # Resampling : every k steps, evaluate and resample
        if step < num_steps - 1 and step % foa_options["k"] == 0:

            # Evaluation : each of the current states is given a value
            value_coroutines = []
            for agent_id, state in enumerate(states):
                value_coroutines.append(GameOf24Agent.evaluate(state, eval_api, namespace=(puzzle_idx, f"Agent: {agent_id}", f"Step : {step}")))
            values = await asyncio.gather(*value_coroutines)

            assert len(eval_api.futures) == 0, f"API futures should be empty, but are {len(eval_api.futures)}"

            # Update records
            for i, (state, value) in enumerate(zip(states, values)):
                state_records.append((f"{step}.{i}", value, state))
            
            # Logging
            log[puzzle_idx][f"Step {step}"]["Evaluation"] = [value for _, value, _ in state_records[-foa_options["num_agents"]:]]
            
            # Resampling
            states, resampled_indices = resampler.resample(state_records, foa_options["num_agents"], foa_options["resampling_method"])
            
            # Logging
            log[puzzle_idx][f"Step {step}"]["Resampling"] = [f"{i} <- {state_records[j][0]}" for i, j in enumerate(resampled_indices)]


    verifications = [GameOf24Agent.verify(result) for result in states]
    log[puzzle_idx]["Input"] = puzzle
    log[puzzle_idx]["Verifications"] = verifications
    return states, log


async def run(run_options: dict, foa_options: dict):
    """
    Inputs
        difficulty: Selects the starting index
        sample_size: Selects the number of experiments to run
    """

    game_coroutines = []
    log = {}

    # Get the data for each puzzle
    puzzle_idxs, puzzles = data.get_data(run_options["set"])

    ### Debugging
    #puzzle_idxs, puzzles = puzzle_idxs[:1], puzzles[:1]

    # Barriers for each puzzle experiment
    barrier = asyncio.Barrier(len(puzzles))

    # Run FoA for each puzzle
    for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles): 
        game_coroutines.append(foa_gameof24(api, limiter, puzzle_idx, puzzle, foa_options, barrier))
    results = await asyncio.gather(*game_coroutines)

    # Merge logs for each run
    logs = [log for (game, log) in results]
    for l in logs:
        log.update(l)
    log["Cost"] = api.cost(verbose=False)

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

    args.add_argument("--set", type=str, choices=["mini", "train", "validation", "test"], default="train")
    args.add_argument("--n_agents", type=int, default=5)
    args.add_argument("--back_coef", type=float, default=0.6)
    args.add_argument("--max_steps", type=int, default=10)
    args.add_argument("--resampling", type=str, choices=["linear", "logistic", "max", "percentile"], default="linear")
    args = args.parse_args()
    return args


args = parse_args()

# Select parameters
set = args.set                             # Set of data to be used
num_agents = args.n_agents                 # Number of agents
k = 1                                      # Resampling every <k> steps
origin_value = 20 * 3                      # The evaluation of the origin state #TODO: Change 3 to num_evaluations
num_steps = args.max_steps                 # Total number of steps FoA executes
backtrack = args.back_coef                 # Backtrack decaying coefficient
resampling_method = args.resampling        # Resampling method


# Just for now so it's easier to change values and reduce noise
log_file = f"{set}-set_{num_agents}agents_{num_steps}steps_{k}k_{origin_value}origin_{backtrack}backtrack_{resampling_method}-resampling.json"
run_options = {
    "set":set
}

foa_options = {
    "num_agents": num_agents,
    "k": k,
    "origin_value": origin_value,
    "num_steps": num_steps,
    "backtrack": backtrack,
    "resampling_method": resampling_method
}

# Run

results = asyncio.run(run(run_options, foa_options))
print(f"File name : {log_file}")
f1 = "logs/2024-03-12/gameof24/22:00/"+log_file
f2 = "logs/same_experiment_logs/"+log_file
same = compare_json_files(f1, f2)
print("Files are the same:", same)

# Accuracy computation
n_success = 0
for game in results:
    verifications = [GameOf24Agent.verify(result) for result in game]
    if {"r": 1} in verifications:
        n_success += 1

# #print accuracy
api.cost(verbose=False)
accuracy = n_success * 100 / len(results)
#print(f"Accuracy : {accuracy:.2f}")

# Send email notification
send_email = False
if send_email:
    subject = log_file
    message = f"Accuracy : {accuracy}"
    try:
        email_notification(subject=subject, message=message)
    except:
        #print("Email not sent")
        pass
