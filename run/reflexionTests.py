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
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from run.reflexion import runReflexionGameOf24, solvePuzzle
from src.agents.reflexionAgent import GameOf24Agent
from src.states.gameof24 import GameOf24State
from utils import load_test_puzzles


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
api = API(eval_api_config, models=models.values(), resources=2, verbose=False)
step_batcher = BatchingAPI(api, batch_size=1, timeout=2, model=models["step"]["model_name"], tab="step")

# Setup logging
logging.basicConfig(
    filename="test_results.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# Suppress logs from external HTTP libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)  # If using OpenAI's SDK


def plotScore(scoreList):
    """
    Plots the number of successfull agents for each reflexion method.
    """
    scores = {
        "list": scoreList,  # Example scores for "list" reflexion
        "k most recent": [4, 5],  # Example scores for "k most recent" reflexion
        "summary_incremental": [6, 7],  # Example scores for "summary" with incremental summary method
        "summary_all_previous": [5, 6]  # Example scores for "summary" with all_previous method
    }
    iterations = [2, 4]  # Example number of iterations
    plt.figure(figsize=(10, 6))

    for method, score in scores.items():
        plt.plot(iterations, score, marker='o', label=method)
    
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
    agent_reflexions = {0:[]}
    finished_puzzles = []
    #TODO: 22212 missing from pickle file for some reason, prbly because it was 
    puzzles = ["1, 1, 4, 6", "1, 1, 11, 11", "1, 3, 8, 8", "1 1 1 8", "6 6 6 6", 
               "1 1 2 12", "1 2 2 6", "1 1 10 12", "2 2 10 10", "1 1 1 12", 
               "3 4 8 12", "2 4 6 11", "2 2 8 9", "1 5 6 7", "5 8 10 11",
               "4 4 9 12", "2 5 6 6", "1 1 3 12", "2 2 2 12", "1 1 4 12"]
    for puzzle in puzzles:
        states, _ = await solvePuzzle(num_steps, puzzle, num_agents, agent_reflexions)
        finished_puzzles.append(states)
    with open("test_puzzles.pkl", "wb") as f:
        pickle.dump(finished_puzzles, f)


async def test_reflexion():
    """
    Test for testing reflexion types
    """
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()
    num_reflexions_list = [2, 4, 8]  # Number of iterations to test
    k = 2  # k for "k most recent"
    num_agents = 4  
    agent_ids = [i for i in range(num_agents)] # Number of active agents
    reflexion_types = ["list", "k most recent", "summary_incremental", "summary_all_previous"]
    results = []

    for states in all_puzzles_data[0:1]:
        for i in range(num_agents):
            states[i] = states[0]
        for num_reflexions in num_reflexions_list:
            for reflexion_type in reflexion_types:
                summary_method = "incremental" if "incremental" in reflexion_type else "all_previous"
                if summary_method == "all_previous":
                    print("all_all previous summary starts now:")
                else:
                    print(reflexion_type, " stars now")
                # Run the reflexion game
                score = await runReflexionGameOf24(
                    states, agent_ids, reflexion_type, num_reflexions, k, summary_method
                )

                # Calculate token cost 
                _, _, token_cost = api.cost(report_tokens=True)

                # Log result
                result_entry = {
                    "puzzle": states[0].puzzle,
                    "num_agents": len(agent_ids),
                    "num_reflexions": num_reflexions,
                    "reflexion_type": reflexion_type,
                    "token_cost": token_cost,
                    "score": score,
                }
                results.append(result_entry)
                logging.info(result_entry)

    # Save results to a pickle file
    with open(f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl", "wb") as f:
        pickle.dump(results, f)

    # Plot the results
    plotScore([r["score"] for r in results if r["reflexion_type"] == "list"])

if __name__ == "__main__":
    asyncio.run(test_reflexion())
