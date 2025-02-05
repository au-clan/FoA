import os
import time
import asyncio
import logging

from copy import deepcopy
from collections import Counter

import openai
import together
from deepdiff import DeepHash
from openai import AsyncAzureOpenAI, AsyncOpenAI
from together import AsyncTogether
from groq import AsyncGroq

from async_engine.round_robin_manager import AsyncRoundRobin
from utils import create_box, merge_responses

logger = logging.getLogger(__name__)


class CachedOpenAIAPI:
    def __init__(self, cache, config, models,
                 resources=1,
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

        # Configure Clients
        ## TODO: Clean this up -> merge clients, limiters and providers into a single dictionary
        self.clients = {}
        self.limiters = {}
        self.providers = {}
        self.groq_requests = 0


        for model in models:

            model_name = model.get("model_name")
            provider = model.get("provider")

            # Save provider for specific model
            self.providers[model_name] = provider

            if provider == "Azure":
                model2keys = {
                    "gpt-4-0613": {"access_token": "AZURE_OPENAI_KEY2LOC1", "endpoint": "key-2"},
                    "gpt-35-turbo-0613": {"access_token": "AZURE_OPENAI_KEY2LOC1", "endpoint": "key-2"},
                    "gpt-35-turbo-0125" : {"access_token":"AZURE_OPENAI_KEY1LOC4", "endpoint": "key-1-18k-loc4"},
                    "gpt-4-0125-preview": {"access_token":"AZURE_OPENAI_KEY2LOC3", "endpoint": "key-2-loc3"},
                    "gpt-4-turbo-2024-04-09": {"access_token": "AZURE_OPENAI_KEY1LOC1", "endpoint": "key-1-18k"},
                    "gpt-4o-2024-05-13": {"access_token": "AZURE_OPENAI_KEY1LOC1", "endpoint": "key-1-18k"},
                    "gpt-4-0613-no-filter": {"access_token": "AZURE_OPENAI_KEY1LOC1", "endpoint": "key-1-18k"},
                    "gpt-4o-2024-05-13-global": {"access_token": "AZURE_OPENAI_KEY1LOC1", "endpoint": "key-1-18k"},
                }

                # Configure access token and endpoint for the model
                access_token = model2keys.get(model_name, {}).get("access_token", False)
                endpoint = model2keys.get(model_name, {}).get("endpoint", False)
                assert bool(access_token and endpoint), f"Model {model_name} not supported!"

                # Get the api key of the model
                api_key = os.getenv(access_token)
                assert api_key, f"Access token '{access_token}' not found in environment variables!"
                print(f"Added access token {access_token} for model {model}.")

                # Client Setup
                self.clients[model_name]= AsyncAzureOpenAI(
                    azure_endpoint="https://"+endpoint+".openai.azure.com/",
                    api_key=api_key,
                    api_version="2024-02-15-preview"
                )

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data=api_key)
            
            elif provider == "Groq":

                # Get the api key for Groq
                access_token = "GROQ_API_KEY"
                api_key = os.getenv(access_token)
                assert api_key, f"Access token '{access_token}' not found in environment variables!"

                #Client Setup
                self.clients[model_name] = AsyncGroq(api_key=api_key)

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data=api_key)
            
            elif provider == "TogetherAI":

                # Get the api key for TogetherAI
                access_token = "TOGETHER_API_KEY_DLAB"
                api_key = os.getenv(access_token)
                assert api_key, f"Access token '{access_token}' not found in environment variables!"

                # Client Setup
                self.clients[model_name] = AsyncTogether(api_key=api_key)

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data=api_key)
            
            elif provider == "OpenAI":
                
                # Get the api key for OpenAI
                access_token = "OPENAI_API_KEY"
                api_key = os.getenv(access_token)
                assert api_key, f"Access token '{access_token}' not found in environment variables!"

                # Client Setup
                self.clients[model_name] = AsyncOpenAI(api_key=access_token)

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data=api_key)
                

            else:
                raise NotImplementedError(f"Provider '{provider}' not supported.\nSupported providers are : Azure, Groq, OpenAI")

        # Save config
        config = deepcopy(config)
        self.config = config

        # Counting tokens to compute cost
        self.tabs = {}

        # Counter
        self.used_count = {}

    async def request(self, messages, namespaces, model: str=None, request_timeout: int=30, tab: str="default"):
        """
        CACHED request to the OpenAI API.
        Input:
        - messages: The prompt
        - namespaces: List of namespace. Each namespace is attributed its own cache entry.
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

        if tab not in self.tabs or model not in self.tabs[tab]:
            self.tabs[tab] = {model:{"completion_tokens": 0, "prompt_tokens": 0, "actual_completion_tokens": 0, "actual_prompt_tokens": 0}}
        
        ##-- Step 1. Setup --##
        responses = []

        # In case user provides the prompt as just a string
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        if "request_timeout" in self.config:
            request_timeout = self.config["request_timeout"]

        # Check if the request is cached
        cache_config = deepcopy(self.config)
        cache_config["model"] = model

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
            if self.verbose:
                print(f"Requesting {n_from_api_total} samples")

            start = time.time()
            async with self.limiters[model] as resource:
                self.clients[model].api_key = resource.data

                if self.providers[model] == "Groq":
                    response = await self.groq_request(model, n_from_api_total, cache_config, request_timeout)
                elif self.providers[model] == "TogetherAI":
                    response = await self.together_request(model, n_from_api_total, cache_config, request_timeout)
                else:
                    response = await self.openai_request(model, n_from_api_total, cache_config, request_timeout)
                # if self.providers[model] in ["Azure", "OpenAI"]:
                #     response = await self.openai_request(model, n_from_api_total, cache_config, request_timeout)
                # else:
                #     response = await self.groq_request(model, n_from_api_total, cache_config, request_timeout)
                raw_responses = [(choice.message.content, response.usage.completion_tokens/n_from_api_total, response.usage.prompt_tokens) for choice in response.choices]

                for retry_count in range(10):
                    if any([r[0] is None for r in raw_responses]):
                        if self.verbose:
                            print(f"Some choices are None, retrying (retry : {retry_count+1})")
                        raw_responses = [r for r in raw_responses if r[0] is not None]
                        n_empty = n_from_api_total - len(raw_responses)
                        if self.providers[model] in ["Azure", "OpenAI"]:
                            response = await self.openai_request(model, n_empty, cache_config, request_timeout)
                        else:
                            response = await self.groq_request(model, n_empty, cache_config, request_timeout)
                        raw_responses.extend([(choice.message.content, response.usage.completion_tokens/n_from_api_total, response.usage.prompt_tokens) for choice in response.choices])

                stop = time.time()
                # by communicating the tokens and time used to the resource manager, we can optimize the rate of requests
                # or maybe we're just using a very simple round robin strategy ;)
                # ToDo: I have a super sophisticated and only slightly buggy implementation of a leaky bucket rate limiting algo
                # for GPT3.5 it works worse than round robin, I think because the rate limit for gpt3.5 are so high that it's
                # easier to just get out of the way and let the next request through
                # but for GPT4 I've found the leaky bucket to be very more efficient
                time_taken = stop - start
                tokens_used = response.usage.total_tokens
                resource.free(time_taken=time_taken, amount_used=tokens_used)

                for r in raw_responses:
                    if r[0] is None:
                        print(create_box("No content in raw response"))
                        for a in response.choices:
                            print(a)
                        
                        exit()
                    

        ##-- Step 4. Requests redistribution --##
        for n, namespace, used_count, cache_entry, cache_key in zip(namespace_counter.values(), namespace_counter.keys(), used_counts, cache_entries, cache_keys):
            
            # Update responses from cache
            n_cache = responses_per_namespace[namespace]["cached"]
            cached_responses = [entry for entry in cache_entry[used_count:used_count+n_cache]]
            self.tabs[tab][model]["completion_tokens"] += sum([entry[1] for entry in cached_responses])
            self.used_count[cache_key] = self.used_count.get(cache_key, 0) + n_cache
            responses.extend([entry[0] for entry in cached_responses])

            # Update responses from API
            n_api = responses_per_namespace[namespace]["api"]
            api_responses = [raw_responses.pop(0) for _ in range(n_api)]
            cache_entry.extend(api_responses)
            self.cache.set(cache_key, cache_entry)
            self.tabs[tab][model]["completion_tokens"] += sum([response[1] for response in api_responses])
            self.tabs[tab][model]["actual_completion_tokens"] += sum([response[1] for response in api_responses]) 
            self.used_count[cache_key] =  self.used_count.get(cache_key, 0) + n_api
            responses.extend([response[0] for response in api_responses])   

        # Pormpt cost
        if n_api != 0:
            self.tabs[tab][model]["prompt_tokens"] += api_responses[0][2]
            self.tabs[tab][model]["actual_prompt_tokens"] += api_responses[0][2]
        elif n_cache != 0:
            self.tabs[tab][model]["prompt_tokens"] += cached_responses[0][2]
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
        
        if self.verbose:
            m = f"+++PROMPT+++\n{messages[0]["content"]}\n\n" + f"+++RESPONSES+++ :\n {"----------------------\n".join([r+"\n" for r in responses])}"
            print("#"*100 + "\n" + m + "\n" + "#"*100)
        return responses



    async def groq_request(self, model, n, cache_config, request_timeout=30):
        responses = []
        
        # Groq currently only allows n=1
        for _ in range(n):
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    self.groq_requests += 1
                    single_response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                    **cache_config,
                    # messages=cache_config["messages"],
                    # model=cache_config["model"],
                    # temperature=cache_config["temperature"],
                    # max_tokens=cache_config["max_tokens"],
                    # timeout=request_timeout,
                    n=1), timeout=request_timeout)
                    assert single_response is not None
                    self.current_sleep_time = self.sleep_time
                    responses.append(single_response)
                    break
                
                except asyncio.TimeoutError as e:
                    print(f"Asyncio timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except Exception as e:
                    if e.response.status_code == 429:
                        print(f"Rate limit error, sleeping for {self.current_sleep_time} seconds")
                        print(e)
                        await asyncio.sleep(self.current_sleep_time)
                        self.current_sleep_time *= self.sleep_factor
                    
                    elif e.response.status_code == 500:
                        print(f"API error, sleeping for {self.current_sleep_time} seconds")
                        print(e)
                        await asyncio.sleep(self.current_sleep_time)
                        self.current_sleep_time *= self.sleep_factor
                    
                    else:
                        print(e)
                        raise e
        
        
        response = merge_responses(responses=responses) # merge_responses(...) counts prompts only once!
        
        assert len(response.choices) == n, f"Expected {n} choices but got {len(response.choices)} choices after merge"
        assert len(responses) == n, f"Expected {n} responses but got {len(responses)} responses" # Assert makes sense as Groq only allows n=1
        
        return response
        

    async def together_request(self, model, n, cache_config, request_timeout=30):
        n_original = n # Used to later verify that all n requests were made
        max_n = 16 # Together AI currently only allows n<=16
        responses = []
        while n > 0:
            n_batch = min(n, max_n)
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    single_response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                        **cache_config,
                        # messages=cache_config["messages"],
                        # model=cache_config["model"],
                        # temperature=cache_config["temperature"],
                        # max_tokens=cache_config["max_tokens"],
                        # timeout=request_timeout,
                        n=n_batch), timeout=request_timeout)
                    
                    assert single_response is not None, "Response is None"
                    self.current_sleep_time = self.sleep_time
                    n -= n_batch
                    responses.append(single_response)
                    break

                except Exception as e:
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except together.error.RateLimitError as e:
                    print(f"Rate limit error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except together.error.InvalidRequestError as e:
                    print(f"Invalid request error\n\n")
                    print(cache_config)
                    print("\n\n")
                    print(e)
                    print("\n\n")
                    print(n)
                    raise e
                
                except together.error.ServiceUnavailableError as e:
                    print(f"Service unavailable error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except together.error.APIStatusError as e:
                    print(f"APIStatusError, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except together.error.APITimeoutError as e:
                    print(f"Timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                

                # except together.error.APIError as e:
                #     print(f"API error, sleeping for {self.current_sleep_time} seconds")
                #     print(e)
                #     await asyncio.sleep(self.current_sleep_time)
                #     self.current_sleep_time *= self.sleep_factor

                except asyncio.TimeoutError as e:
                    print(f"Asyncio timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor
            
        response = merge_responses(responses=responses) # merge_responses(...) counts prompts only once!
        
        assert len(response.choices) == n_original, f"Expected {n_original} choices but got {len(response.choices)} choices after merge"
        
        return response



    async def openai_request(self, model, n, cache_config, request_timeout=30):
        while True:
            self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
            try:
                response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                    **cache_config,
                    # messages=cache_config["messages"],
                    # model=cache_config["model"],
                    # temperature=cache_config["temperature"],
                    # max_tokens=cache_config["max_tokens"],
                    # timeout=request_timeout,
                    n=n), timeout=request_timeout)

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

        assert response is not None, "Response is None"
        return response

    def empty_tabs(self):
        self.tabs = {}

    def cost(self, tab_name=None, actual_cost=False, verbose=False, report_tokens=False):

        # Price catalog
        catalog = {
            "gpt-4-0613": {"prompt_tokens": 0.03, "completion_tokens":0.06},
            "gpt-4-0125-preview": {"prompt_tokens": 0.01, "completion_tokens":0.03},
            "gpt-3.5-turbo-0125": {"prompt_tokens": 0.0005, "completion_tokens":0.0015},
            "gpt-4o-2024-05-13": {"prompt_tokens": 0.005, "completion_tokens":0.015},
            "gpt-4o-2024-05-13-global": {"prompt_tokens": 0.005, "completion_tokens":0.015},
            
            # Llama 3.1 - TogetherAI
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": {"prompt_tokens": 0.88/1000, "completion_tokens":0.88/1000},
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": {"prompt_tokens": 0.18/1000, "completion_tokens":0.18/1000}, 
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt_tokens": 5/1000, "completion_tokens": 5/1000},
            
            # Llama 3.2 - TogetherAI
            "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"  : {"prompt_tokens": 0.18/1000, "completion_tokens":0.18/1000},
            "meta-llama/Llama-Vision-Free"  : {"prompt_tokens": 0.18/1000, "completion_tokens":0.18/1000}, # Llama-3.2-11B
            "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo" : {"prompt_tokens": 1.2/1000, "completion_tokens": 1.2/1000},

            # Mistral - TogetherAI
            "mistralai/Mistral-7B-Instruct-v0.3": {"prompt_tokens": 0.2/1000, "completion_tokens": 0.2/1000}, 
            "mistralai/Mixtral-8x7B-Instruct-v0.1": {"prompt_tokens": 0.6/1000, "completion_tokens": 0.6/1000},
            "mistralai/Mixtral-8x22B-Instruct-v0.1": {"prompt_tokens": 1.2/1000, "completion_tokens": 1.2/1000},

            # Gemma - TogetherAI
            "google/gemma-2-27b-it": {"prompt_tokens": 0.8/1000, "completion_tokens": 0.8/1000},

            "llama" : {"prompt_tokens": 0, "completion_tokens":0},
            "mixtral" : {"prompt_tokens": 0, "completion_tokens":0},
        }

        # Same model just different name
        catalog["gpt-3.5-turbo"] = catalog["gpt-35-turbo-0125"] = catalog["gpt-3.5-turbo-0125"]
        catalog["gpt-4-turbo-2024-04-09"] = catalog["gpt-4-0125-preview"]
        catalog["gpt-4-0613-no-filter"] = catalog["gpt-4-0613"]
        catalog["meta-llama/Meta-Llama-3-8B-Instruct-Turbo"] = catalog["meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]

        input_cost = input_tokens = 0
        output_cost = output_tokens = 0

        # TODO: Clean this up -> No need for initial if else, can be better.
        if tab_name:
            # If a tab is specified, return the cost for the specific tab.
            for model, tokens in self.tabs.get(tab_name, {}).items():
                if actual_cost:
                    input_tokens = tokens["actual_prompt_tokens"]
                    output_tokens = tokens["actual_completion_tokens"]
                else:
                    input_tokens = tokens["prompt_tokens"]
                    output_tokens = tokens["completion_tokens"]
                input_cost += input_tokens / 1000 * catalog[model]["prompt_tokens"]
                output_cost += output_tokens / 1000 * catalog[model]["completion_tokens"]
        else:
            # If no model is specified, return the cost for all models.
            for _, tab in self.tabs.items():
                for model, tokens in tab.items():
                    if actual_cost:
                        input_tokens = tokens["actual_prompt_tokens"]
                        output_tokens = tokens["actual_completion_tokens"]
                    else:
                        input_tokens = tokens["prompt_tokens"]
                        output_tokens = tokens["completion_tokens"]
                    input_cost += input_tokens / 1000 * catalog[model]["prompt_tokens"]
                    output_cost += output_tokens / 1000 * catalog[model]["completion_tokens"]
        
        total_cost = input_cost + output_cost
        total_tokens = input_tokens + output_tokens

        if verbose:
            print(f"Input cost: {input_cost:.3f} USD")
            print(f"Output cost: {output_cost:.3f} USD")
            print(f"Total tokens: {total_cost:.3f} USD")

        if report_tokens:
            return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total_tokens}
        else:
            return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost}