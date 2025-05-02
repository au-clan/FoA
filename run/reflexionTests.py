# Imports
import asyncio
import logging
import math
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

#from src import states
sys.path.append(os.getcwd()) # Project root!!
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from run.reflexion import run_reflexion_gameof24, solve_trial_wise, set_LLMverifier
from src.agents.reflexionAgent import GameOf24Agent
from src.states.gameof24 import GameOf24State
from utils import load_test_puzzles
from src.rafaverifiers import RafaVerifier


step_api_config = eval_api_config = {
    "max_tokens": 1000,
    "temperature": 0,
    "top_p": 1,
    "request_timeout": 120,
    "top_k": 50
}

model = "llama-3.3-70b-versatile"
provider = "LazyKey"
models = {
    "step": {"model_name":model, "provider":provider},
    "eval": {"model_name":model, "provider":provider},
}
api = API(
    eval_api_config, 
    models=models.values(), 
    resources=2, 
    verbose=False
    )

step_batcher = BatchingAPI(
    api, 
    batch_size=1, 
    timeout=2, 
    model=models["step"]["model_name"], 
    tab="step"
    )

# Setup directory for logs
log_dir = "reflexionLogs"

# Setup named loggers for each test type
def setup_logger(name, file_name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(os.path.join(log_dir, file_name))
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger

# Create loggers
trial_logger = setup_logger("trial_wise", "trial_wise.log")
rafa_step_logger = setup_logger("stepwise_RAFA", "stepwise_RAFA.log")
llm_step_logger = setup_logger("stepwise_LLM", "stepwise_LLM.log")

# Suppress logs from external HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)  # If using OpenAI's SDK
logging.getLogger("async_engine").setLevel(logging.WARNING)  # If it's coming from async_engine
logging.getLogger("botocore").setLevel(logging.WARNING)  # If AWS-related

def plotScore(results):
    """
    Plots the number of successful agents for each reflexion method based on test results.
    """
    scores = {}
    
    # Organize scores by reflexion type and number of iterations
    for entry in results:
        method = entry["reflexion_type"]
        num_reflexions = entry["num_reflexions"]
        score = entry["score"]
        
        if method not in scores:
            scores[method] = {}
        scores[method][num_reflexions] = score

    plt.figure(figsize=(10, 6))

    # Sort by iteration count for proper plotting
    num_iterations = sorted({entry["num_reflexions"] for entry in results})

    for method, method_scores in scores.items():
        method_scores_sorted = [method_scores.get(n, 0) for n in num_iterations]
        plt.plot(num_iterations, method_scores_sorted, marker='o', label=method)

    plt.xlabel("Number of Reflexion Iterations")
    plt.ylabel("Score")
    plt.title("Performance of Reflexion Methods in Game of 24")
    plt.legend()
    plt.grid(True)
    plt.show()


async def create_test_puzzles():
    """
    Generates answers for a list of puzzles and saves them to a pickle file
    """
    num_steps = 4
    num_agents = 1
    agent_reflexions = {}
    agent_ids = [i for i in range(num_agents)] # Number of active agents

    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
    
    finished_puzzles = []
    #TODO: 22212 missing from pickle file for some reason, prbly because it was 
    puzzles = ["1, 1, 4, 6", "1, 1, 11, 11", "1, 3, 8, 8", "1 1 1 8", "6 6 6 6", 
               "1 1 2 12", "1 2 2 6", "1 1 10 12", "2 2 10 10", "1 1 1 12", 
               "3 4 8 12", "2 4 6 11", "2 2 8 9", "1 5 6 7", "5 8 10 11",
               "4 4 9 12", "2 5 6 6", "1 1 3 12", "2 2 2 12", "1 1 4 12"]
    puzzles2 = ["1 1 4 6", "1 1 11 11", "6 6 6 6", "1 1 1 12", "1 1 2 12",
                "2 4 7 7", "3 6 6 10", "4 7 9 11", "2 2 3 5", "2 5 7 9",
                "2 4 10 10", "5 5 7 11", "1 3 4 6", "5 7 7 11", "3 3 7 13" ] #5 easy, 5 medium, 5 hard (<99%, 50-52%, 25-27%)
    for puzzle in puzzles2:
        states, _, _ = await solve_trial_wise(step_batcher, num_steps, puzzle, agent_ids, agent_reflexions)
        finished_puzzles.append(states)
    with open("test_puzzles2.pkl", "wb") as f:
        pickle.dump(finished_puzzles, f)


async def trial_wise_type_testing():
    """
    Test for finding the best reflexion type for trial-wise
    """
    print("trial_wise_type_testing")
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    num_reflexions_list = [1,2,4]  # Number of iterations to test
    k = 2  # k for "k most recent"
    num_agents = 4  
    reflexion_types = ["list", "k most recent", "summary_incremental", "summary_all_previous"]   
    results = []
    verifier = RafaVerifier()

    for states in all_puzzles_data[0:15]:
        for i in range(num_agents):
            states[i] = states[0]
        for num_reflexions in num_reflexions_list:
            for reflexion_type in reflexion_types:
                print("puzzle: ", states[0].puzzle, "with type: ", reflexion_type, " starts now")
                # Run the reflexion game
                score, token_cost, num_used_reflexions = await run_reflexion_gameof24(
                    "trial_wise", reflexion_type, states, num_agents, num_reflexions, k, verifier
                )

                # Log result
                result_entry = {
                    "puzzle": states[0].puzzle,
                    "num_agents": num_agents,
                    "num_reflexions": num_reflexions,
                    "reflexion_type": reflexion_type,
                    "score": score,
                    "token_cost": token_cost,
                    "num_used_reflexions": num_used_reflexions
                }
                results.append(result_entry)
                trial_logger.info(result_entry)

async def test_RAFA_stepwise_types():
    """
    Test for step-wise reflexion types with RAFA deterministic verifiers
    """
    print("step_wise type testing with RAFA verifiers")
    set_LLMverifier(False)
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    num_reflexions_list = [1,2,4]  # Number of iterations to test
    k = 1  # k for "k most recent" TODO: Decide if k = 1 would be best here, since last mistake is most relevant for current problem. Should we also change k in trial?
    num_agents = 4  
    reflexion_types = ["list", "k most recent", "summary_incremental", "summary_all_previous"]  # Step-wise reflexion types , "k most recent", "summary_incremental", "summary_all_previous"
    results = []
    verifier = RafaVerifier() 

    for states in all_puzzles_data[0:15]:
        for i in range(num_agents):
            states[i] = states[0]
        for num_reflexions in num_reflexions_list:
            for reflexion_type in reflexion_types:
                print("puzzle: ", states[0].puzzle, "with type: ", reflexion_type, " starts now")
                # Run the step-wise reflexion game
                score, token_cost, num_used_reflexions = await run_reflexion_gameof24(
                    "step_wise", reflexion_type, states, num_agents, num_reflexions, k, verifier
                )

                # Log result
                result_entry = {
                    "puzzle": states[0].puzzle,
                    "num_agents": num_agents,
                    "num_reflexions": num_reflexions,
                    "reflexion_type": reflexion_type,
                    "score": score,
                    "token_cost": token_cost,
                    "num_used_reflexions": num_used_reflexions
                }
                results.append(result_entry)
                rafa_step_logger.info(result_entry)

async def test_LLM_stepwise_reflexion():
    """
    Test for step-wise reflexion types
    """
    print("step_wise type testing with LLM verifiers")
    set_LLMverifier(True)
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    num_reflexions_list = [1,2,4]  # Number of iterations to test
    k = 2  # k for "k most recent"
    num_agents = 4  
    reflexion_types = ["summary_incremental"]  #TODO: Change to the best type determined by RAFA verifiers
    results = []
    verifier = RafaVerifier() #TODO: make rafaVerifier global or smth, it's dumb to have RAFAverifier as a parameter for LLM verifier tests

    for states in all_puzzles_data[0:15]:
        for i in range(num_agents):
            states[i] = states[0]
        for num_reflexions in num_reflexions_list:
            for reflexion_type in reflexion_types:
                print("puzzle: ", states[0].puzzle, "with type: ", reflexion_type, " starts now")
                # Run the step-wise reflexion game
                score, token_cost, num_used_reflexions = await run_reflexion_gameof24(
                    "step_wise", reflexion_type, states, num_agents, num_reflexions, k, verifier
                )

                # Log result
                result_entry = {
                    "puzzle": states[0].puzzle,
                    "num_agents": num_agents,
                    "num_reflexions": num_reflexions,
                    "reflexion_type": reflexion_type,
                    "score": score,
                    "token_cost": token_cost,
                    "num_used_reflexions": num_used_reflexions
                }
                results.append(result_entry)
                llm_step_logger.info(result_entry)

from src.prompts.adapt import gameof24 as llama_prompts

async def scoreTest():
    states = []
    puzzle = "1 1 4 6"
    states.append(GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000)))
    namespace= (0, f"Agent: {int(1)}", f"Step: {1}")
    #prompt = llama_prompts.bfs_prompt_single.format(input=states[0]) 
    prompt = "Hi can you calculate 3 + 2"
    prompt2 = "hello"
    #evaluation = await api.buffered_request(prompt, key=hash(states[0]), namespace=namespace)
    evaluation = await api.request(prompt, namespaces=namespace, model=model, tab = "list"+str(2))
    evaluation = await api.request(prompt2, namespaces=namespace, model=model, tab = "list"+str(4))
    print("evaluation: ", evaluation)
    print(api.cost(tab_name = "list"+str(2), report_tokens=True))
    print(api.cost(tab_name = "list"+str(4), report_tokens=True))
    cost = api.cost(tab_name = "list"+str(2), report_tokens=True)
    token_cost = cost.get("total_tokens")
    print(token_cost)
    cost = api.cost(tab_name = "list"+str(4), report_tokens=True)
    token_cost = cost.get("total_tokens")
    print(token_cost)

if __name__ == "__main__":
    asyncio.run(trial_wise_type_testing())
    asyncio.run(test_RAFA_stepwise_types())
    asyncio.run(test_LLM_stepwise_reflexion())
    #asyncio.run(scoreTest())
    #asyncio.run(create_test_puzzles())
    # with open('test_puzzles.pkl', 'rb') as file:
    #     loaded_list = pickle.load(file)
    # print(loaded_list[0])
    # print(loaded_list[-1])

    
