# Sandbox for testing any implementation

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
from dataclasses import dataclass



from async_engine.cached_api import CachedOpenAIAPI
from async_engine.mock_batched_async import BatchingAPI
from async_engine.round_robin_manager import AsyncRoundRobin

from async_implementation.agents.gameof24 import GameOf24Agent
from async_implementation.states.gameof24 import GameOf24State

from async_implementation.prompts import gameof24 as prompts


@dataclass(frozen=True)
class State:
    prompt: str


class Agent:

    @staticmethod
    async def step(state: State, api: BatchingAPI):
        # make request
        result = await api.buffered_request(state.prompt)

        # do something with the result
        # ...

        return result


async def main():
    # Cache setup
    assert os.path.exists(
        "./caches/"), "Please run the script from the root directory of the project. To make sure all caches are created correctly."
    cache = Cache("./caches/async_api_cache", size_limit=int(2e10))

    # OpenAI API key setup
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    assert OPENAI_API_KEY is not None, "Please set the OPENAI_API_KEY environment variable"

    # API setup
    api_config = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 1,
        "request_timeout": 45,
        "model": "gpt-3.5-turbo"
    }
    api = CachedOpenAIAPI(cache, api_config)

    # Limiter setup
    limiter = AsyncRoundRobin()
    N = 4
    for _ in range(N):
        limiter.add_resource(data=OPENAI_API_KEY)

    prompt = "What is the capital of France?"
    messages = [{"role": "user", "content": prompt}]

    #response = await api.uncached_request(messages, limiter, n=1)
    #print(response)

    # Setup batching API
    batch_size = 1
    bapi = BatchingAPI(api, limiter, batch_size, 10)

    puzzle = "1 1 4 6"
    state = GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000))

    current_state = state.current_state
    prompt = prompts.bfs_prompt.format(input=current_state)

    coroutines = []
    for _ in range(3):
        coroutines.append(bapi.buffered_request(prompt))
    results = await asyncio.gather(*coroutines)
    print(results)

if __name__ == "__main__":
    asyncio.run(main())