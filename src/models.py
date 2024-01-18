import os
from openai import OpenAI, APIConnectionError, APITimeoutError
import backoff
from random import randint

class OpenAIBot:
    def __init__(self, model="gpt-3.5-turbo-1106", temperature=0.5, max_tokens=1000) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise Exception("OPENAI_API_KEY environment variable not found")
        self.client = OpenAI(api_key=api_key)
        self.prompt_tokens = 0
        self.completion_tokens = 0


    @backoff.on_exception(backoff.expo, (APIConnectionError, APITimeoutError))
    def create_completion(self, **kwargs):
        return self.client.chat.completions.create(**kwargs, seed=randint(0,100000))
    

    def request(self, prompt, n=1) -> list:
        messages = [{"role": "user", "content": prompt}]
        response = self.create_completion(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens, n=n)
        output = [response.choices[i].message.content for i in range(n)]
        self.completion_tokens += response.usage.completion_tokens
        self.prompt_tokens += response.usage.prompt_tokens
        return output
    
    
    def print_usage(self):
        input_prices =  {"gpt-4":0.03, "gpt-4-32k":0.06, "gpt-3.5-turbo-1106":0.0010, "gpt-3.5-turbo-instruct":0.0015}
        output_prices = {"gpt-4":0.06, "gpt-4-32k":0.12, "gpt-3.5-turbo-1106":0.0020, "gpt-3.5-turbo-instruct":0.0020}
        if self.model not in input_prices:
            print("No pricing information available for this model")
        else:
            print(f"Input tokens: {self.prompt_tokens} ({self.prompt_tokens / 1000 * input_prices[self.model]} USD)")
            print(f"Output tokens: {self.completion_tokens} ({self.completion_tokens / 1000 * output_prices[self.model]} USD)")
            print(f"Total tokens: {self.prompt_tokens + self.completion_tokens} ({(self.prompt_tokens + self.completion_tokens) / 1000 * (input_prices[self.model] + output_prices[self.model])} USD)")