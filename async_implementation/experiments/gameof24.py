import asyncio
import os

import pandas as pd
from diskcache import Cache

from async_engine.cached_api import CachedOpenAIAPI
from async_engine.round_robin_manager import AsyncRoundRobin
from async_implementation.agents.gameof24 import GameOf24Agent
from async_implementation.states.gameof24 import GameOf24State

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



async def run():
    # set up states
    chosen_puzzle = data[0]
    num_agents = 10
    states = []
    for _ in range(num_agents):
        states.append(GameOf24State(current_state=chosen_puzzle, steps=[]))

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

    # return the final states
    return states


output_states = asyncio.run(run())
