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

async def makeReflexion(puzzle, reflexion_type, num_reflexions, k, states, agent_reflexions, agent_all_reflexions, summary_method):
    step = 3
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(puzzle=puzzle, steps=states[agent_id].steps, state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {int(agent_id)}", f"Step: {step}")
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

async def runReflexionGameOf24(puzzle, agent_ids, typeOfReflexion, num_iterations, k, summary_method="incremental"):
    agent_reflexions = {}
    agent_all_reflexions = {}
    num_steps = 4
    #game_states = []
    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
    #Without reflexion first
    states, agent_ids, total_score = await solvePuzzle(num_steps, puzzle, agent_ids, agent_reflexions)
    #Reflect and go again i times
    for i in range(num_iterations):
        agent_reflexions, agent_all_reflexions = await makeReflexion(puzzle, typeOfReflexion, i+1, k, states, agent_reflexions, agent_all_reflexions, summary_method)
        print("reflexions per agent", agent_reflexions)
        states, agent_ids, score = await solvePuzzle(num_steps, puzzle, agent_ids, agent_reflexions)
        total_score += score
        # game_states.append(states)
    # return game_states
    return total_score
    
    

async def test():
    score = 0
    tests = ["1 1 4 6", "1 1 11 11"]
    num_agents = 4
    agent_ids = [i for i in range(num_agents)]
    num_iterations = [2, 4]
    k = 2
    for puzzle in tests:
        for number_of_iterations in num_iterations:
            #if k < num_iterations:
                #score = 0
                #scoreK = await runReflexionGameOf24(puzzle, agent_ids, "k most recent", number_of_iterations, k)
            scoreList = await runReflexionGameOf24(puzzle, agent_ids, "list", number_of_iterations, k)
            # scoreSummaryIncrement = await runReflexionGameOf24(puzzle, agent_ids, "summary", number_of_iterations, k, "incremental")
            # scoreSummaryAll = await runReflexionGameOf24(puzzle, agent_ids, "summary", number_of_iterations, k, "all_previous")
    plotScore(scoreList)

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



async def main():
    num_iterations = 7
    k = 3
    puzzle = "1 1 4 6"
    num_agents = 4
    agent_ids = [i for i in range(num_agents)]

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
    asyncio.run(test())
