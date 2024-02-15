import numpy as np
import asyncio
import os
import random

import pandas as pd
from diskcache import Cache

from async_engine.cached_api import CachedOpenAIAPI
from async_engine.round_robin_manager import AsyncRoundRobin
from async_implementation.agents.gameof24 import GameOf24Agent
from async_implementation.states.gameof24 import GameOf24State
from async_implementation.resampling import value_weighted

# set up logging
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

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

# set up GameOf24 puzzles
path = 'data/24_tot.csv'
data = pd.read_csv(path).Puzzles.tolist()


# ToDo: this should probably be moved to its own file
# for now I'm keeping it here, for easier debugging
async def foa_gameof24(puzzle_idx:int, num_agents=3, k=2):
    randomness = 0
    random.seed(randomness)

    puzzle = data[puzzle_idx]
    # set up states
    states = []
    for _ in range(num_agents):
        states.append(GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000)))

    num_steps = 10
    for step in range(num_steps):
        # ToDo: eh, log messages are not showing up
        logger.info(f"Step {step}")
        print(f"Step {step}")
        # make one step for each state
        agent_coroutines = []
        for state in states:
            agent_coroutines.append(GameOf24Agent.step(state, api, limiter))
        states = await asyncio.gather(*agent_coroutines)

        # every k steps, evaluate and resample
        if step>0 and step%k==0:
            # evaluate
            value_coroutines = []
            for state in states:
                value_coroutines.append(GameOf24Agent.evaluate(state, api, limiter))
            values = await asyncio.gather(*value_coroutines)

            # resample, ToDo, move this to a function, probably on the agent
            probabilities = value_weighted.linear(values)
            random.seed(randomness)
            randomness = random.randint(0, 1000)
            resampled_indices = np.random.choice(range(len(states)), size=num_agents, p=probabilities, replace=True)

            # this could be dangerous, I'm indexing into the list of states
            # -> the new states will contain ducplicates of the old states
            # but we're using a frozen dataclass to represent states, so duplicates should be fine
            # we can't update the fields in-place in any case
            # any code that wants to mutate state needs to return a new state
            states = [states[i] for i in resampled_indices]





async def run():
    game_coroutines = []
    for idx in range(len(data)):
        # limit to a few games for debugging
        if idx>2:
            break

        game_coroutines.append(foa_gameof24(idx))


    results = await asyncio.gather(*game_coroutines)
    return results


results = asyncio.run(run())
print(results)