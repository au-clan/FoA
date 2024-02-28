import asyncio
import logging
import time
from copy import deepcopy

import openai
from deepdiff import DeepHash
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class CachedOpenAIAPI:
    def __init__(self, cache, config,
                 sleep_time=5,
                 sleep_factor=3,
                 num_retries=3,
                 max_sleep=60,
                 **kwargs):

        self.cache = cache

        # we'll use a resource manager to cycle through keys
        # ideally, the resource manager should ensure that requests are made just below the rate limit
        # in case that doesn't work we also have a more crude sleep mechanism built into the API
        self.sleep_time = sleep_time
        self.sleep_factor = sleep_factor
        self.current_sleep_time = sleep_time
        self.max_sleep = max_sleep
        self.num_retries = num_retries

        config = deepcopy(config)
        self.config = config

        self.aclient = AsyncOpenAI()

        # Counting tokens to compute cost
        self.completion_tokens = 0
        self.prompt_tokens = 0

    async def request(self, messages, limiter, n=10, request_timeout=30):
        """
        CACHED request to the OpenAI API
        """

        if "request_timeout" in self.config:
            request_timeout = self.config["request_timeout"]

        # Check if the request is cached
        cache_config = deepcopy(self.config)

        # we keep only those keys that are relevant for the answer generation
        keep_keys = ["model", "temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "stop"]
        cache_config = {k: v for k, v in cache_config.items() if k in keep_keys}

        # now we also need to add the messages to the cache_config
        cache_config["messages"] = messages

        # NOTE: Updated cache structure to save cost
        # Cache structure {cache_key : cache_entry, ...}
        # Cache entry structure : [(message, completion_tokens, prompt_tokens), ...]

        # use DeepHash to create a key for the cache
        cache_key = DeepHash(cache_config)[cache_config]
        cache_entry = self.cache.get(cache_key)
        if cache_entry is None:
            cache_entry = []

        # the cache_entry is a list of IID responses
        if len(cache_entry) >= n:
            
            # Update tokens count
            self.completion_tokens += sum([entry[1] for entry in cache_entry[:n]])
            self.prompt_tokens += sum([entry[2] for entry in cache_entry[:n]])

            return [entry[0] for entry in cache_entry[:n]]

        # if we don't have enough responses in the cache, we need to make a request
        num_needed = n - len(cache_entry)
        assert num_needed > 0

        print("needing more samples")

        start = time.time()
        async with limiter as resource:
            self.aclient.api_key = resource.data
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    response = await asyncio.wait_for(self.aclient.chat.completions.create(
                        **cache_config,
                        # messages=cache_config["messages"],
                        # model=cache_config["model"],
                        # temperature=cache_config["temperature"],
                        # max_tokens=cache_config["max_tokens"],
                        # timeout=request_timeout,
                        n=num_needed), timeout=request_timeout)

                    self.current_sleep_time = self.sleep_time
                    break
                except openai.RateLimitError as e:
                    print(f"Rate limit error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APIStatusError as e:
                    print(f"APIStatusError, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APITimeoutError as e:
                    print(f"Timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APIError as e:
                    print(f"API error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except asyncio.TimeoutError as e:
                    print(f"Asyncio timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

            assert response is not None

        stop = time.time()
        # by communicating the tokens and time used to the resource manager, we can optimize the rate of requests
        # or maybe we're just using a very simple round robin strategy ;)
        # ToDo: I have a super sophisticated and only slightly buggy implementation of a leaky bucket rate limiting algo
        # for GPT3.5 it works worse than round robin, I think because the rate limit for gpt3.5 are so high that it's
        # easier to just get out of the way and let the next request through
        # but for GPT4 I've found the leaky bucket to be very more efficient
        time_taken = stop - start
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        tokens_used = response.usage.total_tokens
        resource.free(time_taken=time_taken, amount_used=tokens_used)

        logger.log(logging.DEBUG, f"Tokens used: {tokens_used}")

        raw_responses = []
        for choice in response.choices:
            raw_responses.append((choice.message.content, completion_tokens/num_needed, prompt_tokens))

        # add the new responses to the cache
        cache_entry.extend(raw_responses)

        self.cache.set(cache_key, cache_entry)

        # Update tokens count
        self.completion_tokens += sum([entry[1] for entry in cache_entry[:n]])
        self.prompt_tokens += sum([entry[2] for entry in cache_entry[:n]])

        return [entry[0] for entry in cache_entry[:n]]

    async def uncached_request(self, messages, limiter, n=10, request_timeout=30):
        """
        UNCACHED request to the OpenAI API

        Just for debugging purposes
        Same as request(), but without using cache.
        """

        if "request_timeout" in self.config:
            request_timeout = self.config["request_timeout"]

        # Check if the request is cached
        cache_config = deepcopy(self.config)

        # we keep only those keys that are relevant for the answer generation
        keep_keys = ["model", "temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "stop"]
        cache_config = {k: v for k, v in cache_config.items() if k in keep_keys}

        # now we also need to add the messages to the cache_config
        cache_config["messages"] = messages

        # if we don't have enough responses in the cache, we need to make a request
        num_needed = n
        assert num_needed > 0

        print("needing more samples")

        start = time.time()
        async with limiter as resource:
            self.aclient.api_key = resource.data
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    response = await asyncio.wait_for(self.aclient.chat.completions.create(
                        **cache_config,
                        # messages=cache_config["messages"],
                        # model=cache_config["model"],
                        # temperature=cache_config["temperature"],
                        # max_tokens=cache_config["max_tokens"],
                        # timeout=request_timeout,
                        n=num_needed), timeout=request_timeout)

                    self.current_sleep_time = self.sleep_time
                    break
                except openai.RateLimitError as e:
                    print(f"Rate limit error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APIStatusError as e:
                    print(f"APIStatusError, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APITimeoutError as e:
                    print(f"Timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except openai.APIError as e:
                    print(f"API error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except asyncio.TimeoutError as e:
                    print(f"Asyncio timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

            assert response is not None

        stop = time.time()
        # by communicating the tokens and time used to the resource manager, we can optimize the rate of requests
        # or maybe we're just using a very simple round robin strategy ;)
        # ToDo: I have a super sophisticated and only slightly buggy implementation of a leaky bucket rate limiting algo
        # for GPT3.5 it works worse than round robin, I think because the rate limit for gpt3.5 are so high that it's
        # easier to just get out of the way and let the next request through
        # but for GPT4 I've found the leaky bucket to be very more efficient
        time_taken = stop - start
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        tokens_used = response.usage.total_tokens
        resource.free(time_taken=time_taken, amount_used=tokens_used)

        logger.log(logging.DEBUG, f"Tokens used: {tokens_used}")

        raw_responses = []
        for choice in response.choices:
            raw_responses.append((choice.message.content, completion_tokens/num_needed, prompt_tokens))

        # Update tokens count
        self.completion_tokens += sum([entry[1] for entry in raw_responses])
        self.prompt_tokens += sum([entry[2] for entry in raw_responses])

        return [entry[0] for entry in raw_responses[:n]]

    def __str__(self):
        return self.config['model']

    def cost(self, verbose=True):

        # Price catalog
        catalog = {
            "gpt-4": {"prompt_tokens": 0.03, "completion_tokens":0.06},
            "gpt-4-32k": {"prompt_tokens": 0.06, "completion_tokens":0.12},
            "gpt-3.5-turbo-1106": {"prompt_tokens": 0.001, "completion_tokens":0.002},
            "gpt-3.5-turbo-0125": {"prompt_tokens": 0.0005, "completion_tokens":0.0015},
            "gpt-3.5-turbo-instruct": {"prompt_tokens": 0.0015, "completion_tokens":0.002},
        }

        # Name of the model we actually used
        model_used = self.config["model"]

        # "gpt-3.5-turbo" currently (!) refers to "gpt-3.5-turbo-0125"
        if model_used == "gpt-3.5-turbo":
            model_used ="gpt-3.5-turbo-0125"

        if model_used not in catalog:
            print("No pricing information available for this model")
        else:
            input_cost = self.prompt_tokens / 1000 * catalog[model_used]["prompt_tokens"]
            output_cost = self.completion_tokens / 1000 * catalog[model_used]["completion_tokens"]
            total_cost = input_cost + output_cost
            if verbose:
                print(f"Input tokens: {self.prompt_tokens:.0f} ({input_cost:.3f} USD)")
                print(f"Output tokens: {self.completion_tokens:.0f} ({output_cost:.3f} USD)")
                print(f"Total tokens: {self.prompt_tokens + self.completion_tokens:.0f} ({total_cost:.3f} USD)")
            return total_cost