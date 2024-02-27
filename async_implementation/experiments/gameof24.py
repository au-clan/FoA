import asyncio
# set up logging
import logging
import os
import json
import random

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
from async_implementation.agents.gameof24 import GameOf24Agent
from async_implementation.states.gameof24 import GameOf24State
from utils import create_folder, email_notification

logger = logging.getLogger("experiments")
logger.setLevel(logging.DEBUG) # Order : debug < info < warning < error < critical
log_folder = f"logs/{datetime.now().date()}/" # Folder in which logs will be saved (organized daily)
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
    "max_tokens": 1000,
    "temperature": 0.7,
    "top_p": 1,
    "request_timeout": 45,
    "model": "gpt-3.5-turbo-0125"
}

api = CachedOpenAIAPI(cache, api_config)
limiter = AsyncRoundRobin()
# ToDo, this is a bit hacky. OpenAI allows multiple parallel requests per key, so we add the same key multiple times
N = 4
for _ in range(N):
    limiter.add_resource(data=OPENAI_API_KEY)

# set up GameOf24 puzzles
path = 'data/24_tot.csv'
data = pd.read_csv(path).Puzzles.tolist()


# ToDo: this should probably be moved to its own file
# for now I'm keeping it here, for easier debugging
#async def foa_gameof24(puzzle_idx: int, num_agents=3, k=2, backtrack=0.8):
async def foa_gameof24(puzzle_idx, foa_options):
    randomness = 0
    random.seed(randomness)

    puzzle = data[puzzle_idx]
    
    log = {puzzle_idx:{"puzzle": puzzle}}

    # New data structure keeping record of all unique states visited and their according values
    r = {"idx": ["INIT"], "values":[foa_options["origin_value"]], "states": [{"steps":[], "current_state":puzzle}]}

    # set up states
    states = []
    for _ in range(foa_options["num_agents"]):
        states.append(GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000)))

    num_steps = foa_options["num_steps"]
    for step in range(num_steps):
        print(f"Step {step}")
        log[puzzle_idx][f"Step {step}"]={}

        # DONE (?): eh, log messages are not showing up -> Saved them in logs file but not the api ones.
        #logger.info(f"Step: {step}", )

        # make one step for each state
        agent_coroutines = []
        for state in states:
            agent_coroutines.append(GameOf24Agent.step(state, api, limiter))
        states = await asyncio.gather(*agent_coroutines)
        log[puzzle_idx][f"Step {step}"]["steps"] = [" || ".join(state.steps) for state in states]

        # After each step we verify if the answer is found and if so we break
        verifications = [GameOf24Agent.verify(state) for state in states]
        if {"r":1} in verifications:
            break

        # Depreciate old values
        r["values"] = [value * foa_options["backtrack"] if i != 0 else value * (foa_options["backtrack"] ** 2) for i, value in enumerate(r["values"])]

        # every k steps, evaluate and resample
        if step < num_steps -1 and step % foa_options["k"] == 0:
            # evaluate
            value_coroutines = []
            for state in states:
                value_coroutines.append(GameOf24Agent.evaluate(state, api, limiter))
            values = await asyncio.gather(*value_coroutines)

            # Update states tracker
            for i, (state, value) in enumerate(zip(states, values)):
                r["idx"].append(f"{step}.{i}")
                r["values"].append(value)
                r["states"].append({"steps": state.steps, "current_state": state.current_state})
            
            
            # Logging
            log[puzzle_idx][f"Step {step}"]["Evaluation"] = r["values"][-foa_options["num_agents"]:]
            

            # DONE : resample, ToDo, move this to a function, probably on the agent
            resampled_indices = GameOf24Agent.Resampling.linear(r["values"], n_picks=foa_options["num_agents"], randomness=randomness)

            # this could be dangerous, I'm indexing into the list of states
            # -> the new states will contain ducplicates of the old states
            # but we're using a frozen dataclass to represent states, so duplicates should be fine
            # we can't update the fields in-place in any case
            # any code that wants to mutate state needs to return a new state
            resampled_states = [r["states"][i] for i in resampled_indices]
            for i, state in enumerate(resampled_states):
                states[i] = GameOf24State(puzzle=puzzle, current_state=state["current_state"], steps=state["steps"], randomness=random.randint(0, 1000))
            
            # Logging
            log[puzzle_idx][f"Step {step}"]["Resampling"] = [f"{i} <- {r['idx'][j]}" for i, j in enumerate(resampled_indices)]

    verifications = [GameOf24Agent.verify(result) for result in states]
    log[puzzle_idx]["Input"] = puzzle
    log[puzzle_idx]["Verifications"] = verifications
    return states, log


async def run(run_options:dict, foa_options:dict):
    """
    Inputs
        difficulty: Selects the starting index
        sample_size: Selects the number of experiments to run
    """
    game_coroutines = []
    log = {}
    for idx in range(100*run_options["difficulty"], 100*run_options["difficulty"]+run_options["sample_size"]):
        game_coroutines.append(foa_gameof24(idx, foa_options))

    results = await asyncio.gather(*game_coroutines)
    logs = [log for (game, log) in results]
    for l in logs:
        log.update(l)
    with open("logs/full_test.json", 'w+') as f:
        json.dump(log, f, indent=4)
    
    game_states = [game for (game, log) in results]

    return game_states


#################
### Execution ###
#################

# Select parameters
difficulty = 0                  # Starting idx = 100 * difficulty
sample_size = 50                # Ending idx   = 100 * difficulty + sample_size
num_agents = 10                 # Number of agents
k = 1                           # Resampling every <k> steps
origin_value = 20 * 3           # The evaluation of the origin state #TODO: Change 3 to num_evaluations
num_steps = 10                  # Total number of steps FoA executes
backtrack = 0.8                 # Backtrack decaying coefficient


# Just for now so it's easier to change values and reduce noise

run_options = {
    "difficulty":difficulty,
    "sample_size":sample_size}

foa_options = {
    "num_agents":num_agents,
    "k":k,
    "origin_value":origin_value,
    "num_steps":num_steps,
    "backtrack":backtrack
}

# Set handler to log file
handler = logging.FileHandler(log_folder+"full_test.log", mode="w")
logger.addHandler(handler)

results = asyncio.run(run(run_options, foa_options))

n_success = 0
for game in results:
    verifications = [GameOf24Agent.verify(result) for result in game]
    if {"r":1} in verifications:
        n_success+=1
    logger.info(f"Puzzle : {game[0].puzzle}")
    for i, result in enumerate(game):
        logger.info(f"\tAgent {i} current state : {result.current_state}")
        logger.info(f"\t\tSteps : {' -> '.join(result.steps)}")
        logger.info(f"\t\tResult : {bool(verifications[i]['r'])}")
    
    if {"r":1} in verifications:
        logger.info(f"\tGame : Successful\n")
    else:
        logger.info(f"\tGame : Unsuccessful\n")


api.cost(verbose=True)
accuracy = n_success*100/len(results)
print(f"Accuracy : {accuracy:.2f}")
email_notification(subject="0-50, 10 agents, 10 steps", message=f"Accuracy : {accuracy:.2f}")
