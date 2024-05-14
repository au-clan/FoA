import os

import openai
import sys

sys.path.append(os.getcwd()) # Project root!!
from async_implementation.agents.react import WebshopAgent
import async_implementation.prompts.react as prompts
from utils import create_box




# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')


WEBSHOP_URL = 'http://127.0.0.1:3000/abc'
env = WebshopAgent(WEBSHOP_URL)

prompt = prompts.react_prompt
to_print = True

idx = "fixed_0"
action = "reset"
init_prompt = prompt
prompt = ''

try:
    res = env.step(idx, action)
    observation = res[0]
except AssertionError:
    observation = 'Invalid action!'

print(observation)