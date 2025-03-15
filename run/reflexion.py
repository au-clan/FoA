# Imports

from dataclasses import dataclass
from typing import List

import random
import asyncio
import re
import math
import random
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
from dotenv import load_dotenv
import logging
from datetime import datetime

#from src import states


sys.path.append(os.getcwd()) # Project root!!
from async_engine.batched_api import BatchingAPI
from async_engine.api import API
from src.states.gameof24 import GameOf24State
from src.agents.reflexionAgent import GameOf24Agent


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

# Attempting to solve the puzzle
async def solvePuzzle(num_steps, puzzle, agent_ids, agent_reflexions):
    score = 0
    #Create initial state/environment
    states =  {}
    for agent_id in agent_ids:
        states[agent_id] = GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0,1000))
        
    finished_states = {}
    #Stepping
    for step in range(num_steps):
        print(f"Step {step} : Stepping")
        agent_tasks = [
            asyncio.create_task(
            GameOf24Agent.step(states[agent_id], step_batcher, namespace=(0, f"Agent: {agent_id}", f"Step : {step}"), reflexion=agent_reflexions[agent_id])
            )
            for agent_id in states
        ]
        new_states = await asyncio.gather(*agent_tasks)

        for agent_id, new_state in zip(states.keys(), new_states):
            states[agent_id] = new_state
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")

        # Evaluate whether a puzzle has been solved
        for agent_id in list(states.keys()):
            if GameOf24Agent.verify(states[agent_id]) == {"r": 1}:
                print(f"Puzzle finished by agent {agent_id}: {states[agent_id].puzzle}")
                finished_states[agent_id] = states.pop(agent_id)
                agent_ids.remove(agent_id)
                score +=1
            
        # If all puzzles have been solved, break
        if not states:
            break
    return states, agent_ids, score

async def makeReflexion(reflexion_type, k, states, agent_reflexions, agent_all_reflexions, summary_method):
    step = 3
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(puzzle=states[0].puzzle, steps=states[agent_id].steps, state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {int(agent_id)}", f"Step: {step}")
        )
    )
    for agent_id in states
    ] 

    new_reflexions = await asyncio.gather(*agent_tasks)
    for agent_id, reflexion in zip(states.keys(), new_reflexions):
        agent_reflexions[agent_id].append(reflexion)
        agent_all_reflexions[agent_id].append(reflexion)

    if reflexion_type == "list":
        return agent_reflexions, agent_all_reflexions

    elif reflexion_type == "k most recent":
        for agent_id in agent_reflexions:
            agent_reflexions[agent_id] = agent_reflexions[agent_id][-k:]
        return agent_reflexions, agent_all_reflexions

    elif reflexion_type == "summary":
        for agent_id in states:
            print("for agent_id: ", agent_id, "agent_reflexions: ", agent_reflexions[agent_id])
        agent_summaries = []
        if summary_method == "incremental":
            agent_summaries = [
                asyncio.create_task(
                GameOf24Agent.generate_summary(agent_reflexions[agent_id], 
                state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                    )
                )
                for agent_id in states
            ]
        elif summary_method == "all_previous":
            agent_summaries = [
                asyncio.create_task(
                GameOf24Agent.generate_summary(reflexion=agent_all_reflexions[agent_id], 
                state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                    )
                )
                for agent_id in states
            ]
        summaries = await asyncio.gather(*agent_summaries)
        for agent_id, summary in zip(states.keys(), summaries):
            agent_reflexions[agent_id] = [summary] #Replaces reflexions with summary
        return agent_reflexions, agent_all_reflexions
    else:
        print("unknown type")
        return agent_reflexions, agent_all_reflexions

async def runReflexionGameOf24(states, agent_ids, typeOfReflexion, num_iterations, k, summary_method="incremental"):
    puzzle = states[0].puzzle  # Extract puzzle
    agent_reflexions = {}
    agent_all_reflexions = {}
    num_steps = 4
    #game_states = []
    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
    #Without reflexion first, is now done before
    #states, agent_ids, total_score = await solvePuzzle(num_steps, puzzle, agent_ids, agent_reflexions)
    total_score = 0
    #Reflect and go again i times
    for i in range(num_iterations):
        #TODO: change so puzzle not a parameter, but accessed from states
        agent_reflexions, agent_all_reflexions = await makeReflexion(typeOfReflexion, k, states, agent_reflexions, agent_all_reflexions, summary_method)
        print("reflexions per agent", agent_reflexions)
        states, agent_ids, score = await solvePuzzle(num_steps, puzzle, agent_ids, agent_reflexions)
        total_score += score
        # game_states.append(states)
    # return game_states
    return total_score

def plotScore(scoreList):
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

#Are we not missing changing this to agent_ids, and then should we not dump states and agent_ids or how do we access these?
async def create_test_puzzles():
    num_steps = 4
    num_agents = 1
    agent_reflexions = {0:[]}
    finished_puzzles = []
    #TODO: 22212 missing from file for some reason
    puzzles = ["1, 1, 4, 6", "1, 1, 11, 11", "1, 3, 8, 8", "1 1 1 8", "6 6 6 6", 
               "1 1 2 12", "1 2 2 6", "1 1 10 12", "2 2 10 10", "1 1 1 12", 
               "3 4 8 12", "2 4 6 11", "2 2 8 9", "1 5 6 7", "5 8 10 11",
               "4 4 9 12", "2 5 6 6", "1 1 3 12", "2 2 2 12", "1 1 4 12"]
    for puzzle in puzzles:
        states, _ = await solvePuzzle(num_steps, puzzle, num_agents, agent_reflexions)
        finished_puzzles.append(states)
    with open("test_puzzles.pkl", "wb") as f:
        pickle.dump(finished_puzzles, f)

def load_test_puzzles():
    with open("test_puzzles.pkl", "rb") as f:
        puzzles = pickle.load(f)
    return puzzles

async def test():
    tests = ["1 1 4 6", "1 1 11 11"]
    num_agents = 4
    agent_ids = [i for i in range(num_agents)]
    num_iterations = [1,2,3]
    k = 2
    for puzzle in tests:
        for number_of_iterations in num_iterations:
            if k < number_of_iterations:
                scoreK = await runReflexionGameOf24(puzzle, agent_ids, "k most recent", number_of_iterations, k)
            scoreList = await runReflexionGameOf24(puzzle, agent_ids, "list", number_of_iterations, k)
            scoreSummaryIncrement = await runReflexionGameOf24(puzzle, agent_ids, "summary", number_of_iterations, k, "incremental")
            scoreSummaryAll = await runReflexionGameOf24(puzzle, agent_ids, "summary", number_of_iterations, k, "all_previous")
    print(scoreK, scoreList, scoreSummaryIncrement, scoreSummaryAll)

# Setup logging
logging.basicConfig(filename="test_results.log", level=logging.INFO, format="%(asctime)s - %(message)s")

async def test_reflexion():
    # Load unfinished puzzles
    all_puzzles_data = load_test_puzzles()

    num_iterations_list = [2, 4, 8]  # Number of iterations to test
    k = 2  # K for "k most recent"
    num_agents = 4  # Number of agents
    agent_ids = [i for i in range(num_agents)]
    
    reflexion_types = ["list", "k most recent", "summary_incremental", "summary_all_previous"]
    
    results = []

    for states in all_puzzles_data[0:2]:        
        for num_iterations in num_iterations_list:
            for reflexion_type in reflexion_types:
                summary_method = "incremental" if "incremental" in reflexion_type else "all_previous"

                # Run the reflexion game
                score = await runReflexionGameOf24(
                    states, agent_ids, reflexion_type, num_iterations, k, summary_method
                )

                # Calculate token cost 
                _, _, token_cost = api.cost(report_tokens=True)

                # Log result
                result_entry = {
                    "puzzle": states[0].puzzle,
                    "num_agents": len(agent_ids),
                    "num_iterations": num_iterations,
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

async def main():
    num_iterations = 7
    k = 3
    num_agents = 4
    agent_ids = [i for i in range(num_agents)]
    puzzles = load_test_puzzles()
    # print(puzzles)
    puzzle = puzzles[0][0].puzzle #1, 1, 4, 6

    #await runReflexionGameOf24(puzzle, agent_ids, "list", num_iterations, k)
    #await runReflexionGameOf24(puzzle, agent_ids, "k most recent", num_iterations, k)
    await runReflexionGameOf24(puzzle, agent_ids, "summary", num_iterations, k, "incremental")
    #results = await runReflexionGameOf24(puzzle, agent_ids, "summary", num_iterations, k, "incremental")

    #await runReflexionGameOf24(puzzle, agent_ids, "summary", num_iterations, k, "all_previous")
    # n_success = 0
    # for game in results:
    #     verifications = [GameOf24Agent.verify(result) for result in game]
    #     if {"r": 1} in verifications:
    #         n_success += 1
    # accuracy = n_success * 100 / len(results)
    # print(f"Accuracy : {accuracy:.2f}\n")
    

if __name__ == "__main__":
    asyncio.run(test_reflexion())         
