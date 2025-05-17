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
from data.data import GameOf24Data

dataset = GameOf24Data()

step_api_config = eval_api_config = {
    "max_tokens": 1000,
    "temperature": 0.7,
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

def remove_all_retrying_lines(log_path: str):
    # Pattern: Match any line that includes "Retrying request to /openai/v1/chat/completions in X seconds"
    pattern = re.compile(
        r".*Retrying request to /openai/v1/chat/completions in .* seconds.*"
    )

    with open(log_path, 'r') as file:
        lines = file.readlines()

    # Filter out retrying lines
    cleaned_lines = [line for line in lines if not pattern.search(line)]

    with open(log_path, 'w') as file:
        file.writelines(cleaned_lines)

    print(f"Removed all retry lines from {log_path}")


# Create loggers
trial_logger = setup_logger("trial_wise", "trial_wise.log")
rafa_step_logger = setup_logger("stepwise_RAFA", "stepwise_RAFA.log")
llm_step_logger = setup_logger("stepwise_LLM", "stepwise_LLM.log")
k_trial_logger = setup_logger("k_trial_logger", "k_trial.log")
k_step_logger = setup_logger("k_step_logger", "k_step.log")

# Suppress logs from external HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)  # If using OpenAI's SDK
logging.getLogger("async_engine").setLevel(logging.WARNING)  # If it's coming from async_engine
logging.getLogger("botocore").setLevel(logging.WARNING)  # If AWS-related

def plotScore(type_of_reflexions):
    """
    Plots the number of successful agents for each reflexion method based on test type_of_reflexions.
    """
    scores = {}
    
    # Organize scores by reflexion type and number of iterations
    for entry in type_of_reflexions:
        method = entry["reflexion_type"]
        num_reflexions = entry["num_reflexions"]
        score = entry["score"]
        
        if method not in scores:
            scores[method] = {}
        scores[method][num_reflexions] = score

    plt.figure(figsize=(10, 6))

    # Sort by iteration count for proper plotting
    num_iterations = sorted({entry["num_reflexions"] for entry in type_of_reflexions})

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
    puzzle_idxs, puzzles = dataset.get_data("uniform")
    finished_puzzles = []
    # puzzles = ["1, 1, 4, 6", "1, 5, 9, 10", "1, 3, 8, 8", "1 1 1 8", "6 6 6 6"]
    #           "1 1 2 12", "1 2 2 6", "1 1 10 12", "2 2 10 10", "1 1 1 12", 
    #           "3 4 8 12", "2 4 6 11", "2 2 8 9", "1 5 6 7", "5 8 10 11",
    #          "4 4 9 12", "2 5 6 6", "1 1 3 12", "2 2 2 12", "1 1 4 12"]
    puzzles2 = ["1 1 4 6", "1 1 11 11", "6 6 6 6", "1 1 1 12", "1 1 2 12",
                "2 4 7 7", "3 6 6 10", "4 7 9 11", "2 2 3 5", "2 5 7 9",
                "2 4 10 10", "5 5 7 11", "1 3 4 6", "5 7 7 11", "3 3 7 13" ] #5 easy, 5 medium, 5 hard (<99%, 50.3-52.7%, 25.8-27.6%)
    for puzzle in puzzles:
        agent_id = [0]
        states, _, _ = await solve_trial_wise(step_batcher, num_steps, puzzle_idxs, puzzle, agent_id, agent_reflexions, {})
        finished_puzzles.append(states)
        print("agent_ids", agent_ids)
        print("State in create test puzzles:", states)
    print(len(finished_puzzles))
    with open("uniform_test_puzzles.pkl", "wb") as f:
        pickle.dump(finished_puzzles, f)

async def run_puzzles(
    time_of_reflexion: str,
    puzzle_idx, 
    states,
    logger):
    num_reflexions_list = [1]  # Number of iterations to test ,2,4
    k = 2  # k for "k_most_recent"
    num_agents = 1  
    reflexion_types = ["list", "k_most_recent", "summary_incremental", "summary_all_previous"]   
    type_of_reflexions = []

    for i in range(num_agents):
        states[i] = states[0]
    for num_reflexions in num_reflexions_list:
        for reflexion_type in reflexion_types:
            print("\npuzzle: ", states[0].puzzle, "with type: ", reflexion_type, " starts now")
            # Run the reflexion game
            score, tokens_used, total_tokens, num_used_reflexions = await run_reflexion_gameof24(
                time_of_reflexion, reflexion_type, int(puzzle_idx), states, num_agents, num_reflexions, k
            )

            # Log type_of_reflexion
            type_of_reflexion_entry = {
                "puzzle": states[0].puzzle,
                "num_agents": num_agents,
                "num_reflexions": num_reflexions,
                "reflexion_type": reflexion_type,
                "score": score,
                "tokens_used": tokens_used,
                "total_tokens": total_tokens,
                "num_used_reflexions": num_used_reflexions
            }
            logger.info(type_of_reflexion_entry)
            type_of_reflexions.append(type_of_reflexion_entry)

    return type_of_reflexions

async def find_k(
    time_of_reflexion: str,
    puzzle_idx, 
    states,
    logger):
    num_reflexions = 1  # Number of iterations of reflexion 4
    ks = [1]  # k for "k_most_recent",2,4
    num_agents = 1  
    reflexion_type = "k_most_recent"
    type_of_reflexions = []

    for i in range(num_agents):
        states[i] = states[0]
    for k in ks:
        print("\npuzzle: ", states[0].puzzle, "with type: ", reflexion_type, " starts now")
        # Run the reflexion game
        score, tokens_used, total_tokens, num_used_reflexions = await run_reflexion_gameof24(
            time_of_reflexion, reflexion_type, int(puzzle_idx), states, num_agents, num_reflexions, k
        )

        # Log type_of_reflexion
        type_of_reflexion_entry = {
            "puzzle": states[0].puzzle,
            "num_agents": num_agents,
            "num_reflexions": num_reflexions,
            "reflexion_type": reflexion_type,
            "k": k,
            "score": score,
            "tokens_used": tokens_used,
            "total_tokens": total_tokens,
            "num_used_reflexions": num_used_reflexions
        }
        logger.info(type_of_reflexion_entry)
        type_of_reflexions.append(type_of_reflexion_entry)

    return type_of_reflexions

async def trial_wise_type_testing():
    """
    Test for finding the best reflexion type for trial_wise
    """
    print("\ntrial_wise_type_testing")
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    puzzle_idxs, _ = dataset.get_data("uniform")
    tasks = [
        asyncio.create_task(
        run_puzzles(
            "trial_wise",
            puzzle_idx=puzzle_idxs[idx],
            states = all_puzzles_data[idx],
            logger=trial_logger
            )
        )
        for idx in range(0,1)
        #for idx in range(min(len(puzzle_idxs), len(all_puzzles_data)))
    ]
    all_type_of_reflexions = await asyncio.gather(*tasks)

async def test_RAFA_stepwise_types():
    """
    Test for step_wise reflexion types with RAFA deterministic verifiers
    """
    print("\nstep_wise type testing with RAFA verifiers")
    set_LLMverifier(False)
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    puzzle_idxs, _ = dataset.get_data("uniform")

    tasks = [
        asyncio.create_task(
        run_puzzles(
            "step_wise",
            puzzle_idx=puzzle_idxs[idx],
            states = all_puzzles_data[idx],
            logger=rafa_step_logger
            )
        )
        for idx in range(0,1)
        #for idx in range(min(len(puzzle_idxs), len(all_puzzles_data)))
    ]

    all_type_of_reflexions = await asyncio.gather(*tasks)

async def test_LLM_stepwise_reflexion():
    """
    Test for step_wise reflexion types
    """
    print("\nstep_wise type testing with LLM verifiers")
    set_LLMverifier(True)
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    puzzle_idxs, _ = dataset.get_data("uniform")

    tasks = [
        asyncio.create_task(
        run_puzzles(
            "step_wise",
            puzzle_idx=puzzle_idxs[idx],
            states = all_puzzles_data[idx],
            logger=llm_step_logger
            )
        )
        for idx in range(0,1)
        #or idx in range(min(len(puzzle_idxs), len(all_puzzles_data)))
    ]

    all_type_of_reflexions = await asyncio.gather(*tasks)

from src.prompts.adapt import gameof24 as llama_prompts

async def test_K_value():
    """
    test for finding the best k value in both trial_wise and step_wise
    """
    print("\nK testing starts now")
    set_LLMverifier(False)
    all_puzzles_data = load_test_puzzles()
    puzzle_idxs, _ = dataset.get_data("uniform")

    trial_tasks = [
        asyncio.create_task(
        find_k(
            "trial_wise",
            puzzle_idx=puzzle_idxs[idx],
            states = all_puzzles_data[idx],
            logger=k_trial_logger
            )
        )    
        for idx in range(0,1)
        #for idx in range(min(len(puzzle_idxs), len(all_puzzles_data)))
    ]

    trial_type_of_reflexions = await asyncio.gather(*trial_tasks)

    step_tasks = [
        asyncio.create_task(
        find_k(
            "step_wise",
            puzzle_idx=puzzle_idxs[idx],
            states = all_puzzles_data[idx],
            logger=k_step_logger
            )
        )    
        for idx in range(0,1)
        #for idx in range(min(len(puzzle_idxs), len(all_puzzles_data)))
    ]

    step_type_of_reflexions = await asyncio.gather(*step_tasks)

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
    asyncio.run(test_LLM_stepwise_reflexion())
    asyncio.run(test_RAFA_stepwise_types())
    asyncio.run(test_K_value())
    # asyncio.run(trial_wise_type_testing())
    # asyncio.run(test_RAFA_stepwise_types())
    #asyncio.run(test_LLM_stepwise_reflexion())
    #asyncio.run(scoreTest())
    # asyncio.run(create_test_puzzles())
    # with open('uniform_test_puzzles.pkl', 'rb') as file:
    #     loaded_list = pickle.load(file)
    # for i in range(len(loaded_list)):
    #     # print(loaded_list[i])
    #     if len(loaded_list[i]) == 0:
    #         print(i)
    # print(len(loaded_list))
    # idxs, puzzles = dataset.get_data("uniform")
    # print(idxs)
    # print(puzzles)
    # print(len(puzzles))
    
    
    #remove_all_retrying_lines("reflexionLogs/failedTrialwise2.log")
