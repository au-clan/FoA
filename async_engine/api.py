import os
import time
import asyncio
import traceback
from dotenv import load_dotenv

from copy import deepcopy
from collections import Counter

import openai
import together
from openai import AsyncOpenAI
from together import AsyncTogether
from groq import AsyncGroq
from lazykey import AsyncKeyHandler

from async_engine.round_robin_manager import AsyncRoundRobin
from utils import merge_responses

class API:
    def __init__(self, config, models,
                 resources=1,
                 sleep_time=5,
                 sleep_factor=3,
                 num_retries=3,
                 max_sleep=60,
                 verbose=True,
                 **kwargs):

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
        self.lazykey_requests = 0


        for model in models:

            model_name = model.get("model_name")
            provider = model.get("provider")

            # Save provider for specific model
            self.providers[model_name] = provider
            
            if provider == "Groq":

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
                access_token = "TOGETHER_API_KEY"
                api_key = os.getenv(access_token)
                assert api_key, f"Access token '{access_token}' not found in environment variables!"

                # Client Setup
                self.clients[model_name] = AsyncTogether(api_key=api_key)

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data="api_key")
            
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
                
            elif provider == "LazyKey":
                #api keys when making an env
                # load_dotenv()
                # api_keys = os.environ.get("API_KEYS")
                api_keys = ["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"]
                self.clients[model_name] = AsyncKeyHandler(api_keys, AsyncGroq)

                # Limiter Setup
                self.limiters[model_name] = AsyncRoundRobin()
                for _ in range(resources):
                    self.limiters[model_name].add_resource(data="api_key")
            else:
                raise NotImplementedError(f"Provider '{provider}' not supported.\nSupported providers are : Groq, TogetherAI and OpenAI")

        # Save config
        config = deepcopy(config)
        self.config = config

        # Counting tokens to compute cost
        self.tabs = {}

        # Counter
        self.used_count = {}

    async def request(self, messages, namespaces, model: str=None, request_timeout: int=30, tab: str="default"):
        """
        Input:
        - messages: The prompt
        - namespaces: List of namespace. Each namespace is attributed its own cache entry.
        - request_timeout: Timeout for the request
        - tab: The tab for which the request is made

        Tabs (self.tabs) : Whoever calls this function can specify a tab. The API tracks the cost for each tab independently.
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
        request_config = deepcopy(self.config)
        request_config["model"] = model

        # we keep only those keys that are relevant for the answer generation
        keep_keys = ["model", "temperature", "max_tokens", "top_p", "presence_penalty", "frequency_penalty", "stop"]
        request_config = {k: v for k, v in request_config.items() if k in keep_keys}

        # now we also need to add the messages to the CACHE_CONFIG
        request_config["messages"] = messages


        n = len(namespaces)
        if n > 0:
            if self.verbose:
                print(f"Requesting {n} samples")

            start = time.time()
            async with self.limiters[model] as resource:
                self.clients[model].api_key = resource.data

                if self.providers[model] == "Groq":
                    response = await self.groq_request(model, n, request_config, request_timeout)
                elif self.providers[model] == "TogetherAI":
                    response = await self.together_request(model, n, request_config, request_timeout)
                elif self.providers[model] == "LazyKey":
                    response = await self.lazykey_request(model, n, request_config, request_timeout)
                else:
                    response = await self.openai_request(model, n, request_config, request_timeout)

                raw_responses = [(choice.message.content, response.usage.completion_tokens/n, response.usage.prompt_tokens) for choice in response.choices]

                stop = time.time()

                time_taken = stop - start
                tokens_used = response.usage.total_tokens
                resource.free(time_taken=time_taken, amount_used=tokens_used)
        
        self.tabs[tab][model]["completion_tokens"] += sum([response[1] for response in raw_responses])
        self.tabs[tab][model]["prompt_tokens"] += raw_responses[0][2]
        responses = [response[0] for response in raw_responses]

        
        ##-- Step 5. Rearrange responses --##
        initial_responses = responses.copy()
        mapping = {}
        namespace_counter = Counter(namespaces)
        for k,v in namespace_counter.items():
            mapping[k] = []
            for i in range(v):
                mapping[k].append(responses.pop(0))
        assert responses == []

        for namespace in namespaces:
            responses.append(mapping[namespace].pop(0))
        
        if self.verbose:
            pass
            #m = f"+++PROMPT+++\n{messages[0]["content"]}\n\n" + f"+++RESPONSES+++ :\n {"----------------------\n".join([r+"\n" for r in responses])}"
            #print("#"*100 + "\n" + m + "\n" + "#"*100)
        return responses



    async def groq_request(self, model, n, CACHE_CONFIG, request_timeout=30):
        responses = []
        
        # Groq currently only allows n=1
        for _ in range(n):
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    self.groq_requests += 1
                    single_response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                    **CACHE_CONFIG,
                    # messages=CACHE_CONFIG["messages"],
                    # model=CACHE_CONFIG["model"],
                    # temperature=CACHE_CONFIG["temperature"],
                    # max_tokens=CACHE_CONFIG["max_tokens"],
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

    async def lazykey_request(self, model, n, CACHE_CONFIG, request_timeout=30):
        responses = []
        
        # Groq currently only allows n=1
        for _ in range(n):
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                
                    self.lazykey_requests += 1
                    single_response = await asyncio.wait_for(self.clients[model].request(
                    **CACHE_CONFIG,
                    # messages=CACHE_CONFIG["messages"],
                    # model=CACHE_CONFIG["model"],
                    # temperature=CACHE_CONFIG["temperature"],
                    # max_tokens=CACHE_CONFIG["max_tokens"],
                    # timeout=request_timeout,
                    n=1), timeout=request_timeout)
                    assert single_response is not None
                    self.current_sleep_time = self.sleep_time
                    responses.append(single_response)
                    break
                
                except asyncio.TimeoutError as e:
                    traceback.print_exc()
                    print(f"Asyncio timeout error, sleeping for {self.current_sleep_time} seconds")
                    print(e)
                    await asyncio.sleep(self.current_sleep_time)
                    self.current_sleep_time *= self.sleep_factor

                except Exception as e:
                    if e.response.status_code == 429:
                        traceback.print_exc()
                        print(f"Rate limit error, sleeping for {self.current_sleep_time} seconds")
                        print(e)
                        await asyncio.sleep(self.current_sleep_time)
                        self.current_sleep_time *= self.sleep_factor
                    
                    elif e.response.status_code == 500:
                        traceback.print_exc()
                        print(f"API error, sleeping for {self.current_sleep_time} seconds")
                        print(e)
                        await asyncio.sleep(self.current_sleep_time)
                        self.current_sleep_time *= self.sleep_factor
                    
                    else:
                        traceback.print_exc()
                        print(e)
                        raise e
        
        
        response = merge_responses(responses=responses) # merge_responses(...) counts prompts only once!
        
        assert len(response.choices) == n, f"Expected {n} choices but got {len(response.choices)} choices after merge"
        assert len(responses) == n, f"Expected {n} responses but got {len(responses)} responses" # Assert makes sense as Groq only allows n=1
        
        return response
        

    async def together_request(self, model, n, CACHE_CONFIG, request_timeout=30):
        n_original = n # Used to later verify that all n requests were made
        max_n = 16 # Together AI currently only allows n<=16
        responses = []
        while n > 0:
            n_batch = min(n, max_n)
            while True:
                self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
                try:
                    single_response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                        **CACHE_CONFIG,
                        # messages=CACHE_CONFIG["messages"],
                        # model=CACHE_CONFIG["model"],
                        # temperature=CACHE_CONFIG["temperature"],
                        # max_tokens=CACHE_CONFIG["max_tokens"],
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
                    print(CACHE_CONFIG)
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



    async def openai_request(self, model, n, CACHE_CONFIG, request_timeout=30):
        while True:
            self.current_sleep_time = min(self.current_sleep_time, self.max_sleep)
            try:
                response = await asyncio.wait_for(self.clients[model].chat.completions.create(
                    **CACHE_CONFIG,
                    # messages=CACHE_CONFIG["messages"],
                    # model=CACHE_CONFIG["model"],
                    # temperature=CACHE_CONFIG["temperature"],
                    # max_tokens=CACHE_CONFIG["max_tokens"],
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

            #DeepSeek - Groq
            "deepseek-r1-distill-llama-70b": {"prompt_tokens": 0.8/1000, "completion_tokens": 0.8/1000},
            "llama-3.2-11b-vision-preview": {"prompt_tokens": 0.8/1000, "completion_tokens": 0.8/1000},
            "llama-3.3-70b-versatile": {"prompt_tokens": 0.8/1000, "completion_tokens": 0.8/1000},
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