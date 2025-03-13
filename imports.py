# Imports
#from lazykey import AsyncKeyHandler
from dataclasses import dataclass
from typing import List

import random

import asyncio
import re
import math
import random
import numpy as np
from sympy import simplify
import os
# importing necessary functions from dotenv library
import os
#from dotenv import load_dotenv

random.seed(0)

from async_engine.batched_api import BatchingAPI
from async_engine.api import API
from groq import AsyncGroq

from src.prompts.adapt import gameof24 as llama_prompts
from utils import parse_suggestions, create_box


def main():
    print("hej")

if __name__ == "__main__":
    main()