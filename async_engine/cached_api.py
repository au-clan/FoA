import asyncio
import logging
import time
from copy import deepcopy
from collections import Counter

import openai
from deepdiff import DeepHash
from openai import AsyncAzureOpenAI

from utils import create_box

logger = logging.getLogger(__name__)


class CachedOpenAIAPI:
    def __init__(self, cache, config,
                 sleep_time=5,
                 sleep_factor=3,
                 num_retries=3,
                 max_sleep=60,
                 verbose=True,
                 **kwargs):

        self.cache = cache
        self.verbose = verbose

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

        self.aclient = AsyncAzureOpenAI(azure_endpoint="https://key-2-loc2.openai.azure.com/", api_version="2024-02-15-preview")
        # Counting tokens to compute cost
        self.tabs = {}

        # Counter
        self.used_count = {}

    async def request(self, messages, namespaces, limiter, request_timeout: int=30, tab: str="default"):
        """
        CACHED request to the OpenAI API.
        Input:
        - messages: The prompt
        - namespaces: List of namespace. Each namespace is attributed its own cache entry.
        - limiter: Resource manager
        - request_timeout: Timeout for the request
        - tab: The tab for which the request is made

        Tabs (self.tabs) : Whoever calls this function can specify a tab. The API tracks the cost for each tab independently.
        
        The function is executed in the following steps:
        Step 1. Setup : 
            - Sets up the cache config for each namesapce and gets the respective cache key, cahce entry and used count.
        Step 2. Prepare number of requests:
            - Seperates the number of total requests needed for each namespace to number of requests from cache and number of requests from API.
        Step 3. Request from API:
            - Requests the total number of requests (no matter the namespace) from the API into responses.
        Step 4. Requests redistribution:
            - For each namespace, according to the computed number of requests needed from the cache, load the number of requests needed from the cache and update the cache entry and used count.
            - For each namespace, according to the computed number of requests needed from the API, pop the number of responses needed from the API responses and update the cache entry and used count.
        Step 5. Rearrange responses:
            - Rearranges the responses to match the order of the input namespace list.
        """

        if tab not in self.tabs:
            self.tabs[tab] = {"completion_tokens": 0, "prompt_tokens": 0, "actual_completion_tokens": 0, "actual_prompt_tokens": 0}
        
        ##-- Step 1. Setup --##
        responses = []
        messages = [{"role": "user", "content": messages}]
        if "request_timeout" in self.config:
            request_timeout = self.config["request_timeout"]

        # Check if the request is cached
        cache_config = deepcopy(self.config)

        # we keep only those keys that are relevant for the answer generation
        keep_keys = ["model", "temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "stop"]
        cache_config = {k: v for k, v in cache_config.items() if k in keep_keys}

        # now we also need to add the messages to the cache_config
        cache_config["messages"] = messages

        namespace_counter = Counter(namespaces)
        responses_per_namespace = {}

        cache_configs = []
        for namespace in namespace_counter.keys():
            temp_cache_config = deepcopy(cache_config)
            temp_cache_config["namespace"] = namespace
            cache_configs.append(temp_cache_config)

        # NOTE: Updated cache structure to save cost
        # Cache structure {cache_key : cache_entry, ...}
        # Cache entry structure : [(message, completion_tokens, prompt_tokens), ...]

        # use DeepHash to create a key for the cache
        cache_keys = [DeepHash(cc)[cc] for cc in cache_configs]
        cache_entries = []
        for cache_key in cache_keys:
            cache_entry = self.cache.get(cache_key)
            if cache_entry is None:
                cache_entry = []
            cache_entries.append(cache_entry)

        # Get the number of times this cached entry has been used and update the count
        used_counts = [self.used_count.get(cache_key, 0) for cache_key in cache_keys]
        
        # Unrelated stuff to be fixed if we want to use this  
        # -- Does not consider prompt cost
        
        ##-- Step 2. Prepare number of requests --##
        # Compute the number of samples needed from the cache and from the API
        for n, namespace, used_count, cache_entry, cache_key in zip(namespace_counter.values(), namespace_counter.keys(), used_counts, cache_entries, cache_keys):
            cached_available = len(cache_entry[used_count:])
            n_from_cache = min(n, cached_available)
            n_from_api = n - n_from_cache
            assert n_from_api >= 0

            responses_per_namespace[namespace] = {"cached": n_from_cache, "api": n_from_api}
        
        n_from_api_total = sum(responses_per_namespace[namespace]["api"] for namespace in responses_per_namespace)
        n_from_cache_total = sum(responses_per_namespace[namespace]["cached"] for namespace in responses_per_namespace)

        if n_from_cache_total > 0 and self.verbose:
            print(f"Using {n_from_cache_total} cached samples")

        ##-- Step 3. Request from API --##
        # Request the samples from the API
        if n_from_api_total > 0:
            print(f"Requesting {n_from_api_total} samples")

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
                            n=n_from_api_total), timeout=request_timeout)

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

                raw_responses = []
                for choice in response.choices:
                    raw_responses.append((choice.message.content, completion_tokens/n_from_api_total, prompt_tokens))
                    
                if any(response[0]==None for response in raw_responses):
                    for choice in response.choices:
                        print(create_box(choice))
                    assert False, "None response in raw_responses"

        ##-- Step 4. Requests redistribution --##
        for n, namespace, used_count, cache_entry, cache_key in zip(namespace_counter.values(), namespace_counter.keys(), used_counts, cache_entries, cache_keys):
            
            # Update responses from cache
            n_cache = responses_per_namespace[namespace]["cached"]
            cached_responses = [entry for entry in cache_entry[used_count:used_count+n_cache]]
            self.tabs[tab]["completion_tokens"] += sum([entry[1] for entry in cached_responses])
            self.used_count[cache_key] = self.used_count.get(cache_key, 0) + n_cache
            responses.extend([entry[0] for entry in cached_responses])

            # Update responses from API
            n_api = responses_per_namespace[namespace]["api"]
            api_responses = [raw_responses.pop(0) for _ in range(n_api)]
            cache_entry.extend(api_responses)
            self.cache.set(cache_key, cache_entry)
            self.tabs[tab]["completion_tokens"] += sum([response[1] for response in api_responses])
            self.tabs[tab]["actual_completion_tokens"] += sum([response[1] for response in api_responses]) 
            self.used_count[cache_key] =  self.used_count.get(cache_key, 0) + n_api
            responses.extend([response[0] for response in api_responses])   

        # Pormpt cost
        if n_api != 0:
            self.tabs[tab]["prompt_tokens"] += api_responses[0][2]
            self.tabs[tab]["actual_prompt_tokens"] += api_responses[0][2]
        elif n_cache != 0:
            self.tabs[tab]["prompt_tokens"] += cached_responses[0][2]
        else:
            raise Exception("Both n_api and n_cache are zero. This should not happen.")
        
        ##-- Step 5. Rearrange responses --##
        initial_responses = responses.copy()
        mapping = {}
        for k,v in namespace_counter.items():
            mapping[k] = []
            for i in range(v):
                mapping[k].append(responses.pop(0))
        assert responses == []

        for namespace in namespaces:
            responses.append(mapping[namespace].pop(0))

        if any([response== None for response in responses]):
            for i, response in enumerate(responses):
                error = f"Response:\n{response}\nInitial responses:\n{initial_responses[i]}\n"
                print(create_box(error))

            assert False, "None response in responses"
        return responses


    def __str__(self):
        return self.config['model']

    def cost(self, actual_cost=False, tab=None, verbose=False):

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
        if model_used == "gpt-3.5-turbo" or "gpt-35-turbo-0125":
            model_used ="gpt-3.5-turbo-0125"

        if model_used not in catalog:
            print("No pricing information available for this model")
        else:
            if tab:
                # If a tab is specified, return the cost for the specific tab.
                if actual_cost:
                    input_tokens = self.tabs.get(tab, {}).get("actual_prompt_tokens", 0)
                    output_tokens = self.tabs.get(tab, {}).get("actual_completion_tokens", 0)
                else:
                    input_tokens = self.tabs.get(tab, {}).get("prompt_tokens", 0)
                    output_tokens = self.tabs.get(tab, {}).get("completion_tokens", 0)
            else:
                # If no tab is specified, return the cost for all tabs.
                if actual_cost:
                    input_tokens = sum([tab["actual_prompt_tokens"] for tab in self.tabs.values()])
                    output_tokens = sum([tab["actual_completion_tokens"] for tab in self.tabs.values()])
                else:
                    input_tokens = sum([tab["prompt_tokens"] for tab in self.tabs.values()])
                    output_tokens = sum([tab["completion_tokens"] for tab in self.tabs.values()])

            input_cost = input_tokens / 1000 * catalog[model_used]["prompt_tokens"]
            output_cost = output_tokens / 1000 * catalog[model_used]["completion_tokens"]
            total_cost = input_cost + output_cost
            if verbose:
                print(f"Input tokens: {input_tokens:.0f} ({input_cost:.3f} USD)")
                print(f"Output tokens: {output_tokens:.0f} ({output_cost:.3f} USD)")
                print(f"Total tokens: {input_tokens + output_tokens:.0f} ({total_cost:.3f} USD)")
            return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_cost": total_cost}