# Imports
import asyncio
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
sys.path.append(os.getcwd()) # Project root!!
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
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
api = API(
    eval_api_config, 
    models=models.values(), 
    resources=2, 
    verbose=False
    )
    
step_batcher = BatchingAPI(
    api, batch_size=1, 
    timeout=2, 
    model=models["step"]["model_name"], 
    tab="step"
    )


async def solve_puzzle(num_steps: int, puzzle: str, agent_ids: List[int], agent_reflexions: Dict[int, List[str]]) -> Tuple[Dict[int, GameOf24State], List[int], int]:
    """"
    Solves the puzzle either with or without reflections.
    Returns the updated states, remaining agent_ids, and the accumulated score.
    """
    score = 0
    states =  {}
    #Create one state for each agent
    for agent_id in agent_ids:
        states[agent_id] = GameOf24State(
            puzzle=puzzle, 
            current_state=puzzle, 
            steps=[], 
            randomness=random.randint(0,1000)
            )
    finished_states = {}

    #Stepping
    for step in range(num_steps):
        print(f"Step {step} : Stepping")
        agent_tasks = [
            asyncio.create_task(
            GameOf24Agent.step(
                states[agent_id], 
                step_batcher, 
                namespace=(0, f"Agent: {agent_id}", f"Step : {step}"), #TODO: fix namespace stuff
                reflexion=agent_reflexions[agent_id])
            )
            for agent_id in states
        ]
        new_states = await asyncio.gather(*agent_tasks)

        for agent_id, new_state in zip(states.keys(), new_states):
            states[agent_id] = new_state
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")

        # Evaluate whether a puzzle has been solved, 
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


async def make_reflexion(
    reflexion_type: str, 
    k: int, 
    states: Dict[int, GameOf24State], 
    agent_reflexions: Dict[int, List[str]], 
    agent_all_reflexions: Dict[int, List[str]], 
    summary_method: str
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Generates a reflection for each agent based on their current state and the chosen type of reflection.
    """
    step = 3
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(
                puzzle=states[0].puzzle, 
                steps=states[agent_id].steps, 
                state=states[agent_id], 
                api=step_batcher, 
                namespace=(0, f"Agent: {int(agent_id)}", 
                f"Step: {step}")
        )
    ) for agent_id in states ]
    new_reflexions = await asyncio.gather(*agent_tasks)
    
    for agent_id, reflexion in zip(states.keys(), new_reflexions):
        agent_reflexions[agent_id].append(reflexion)
        agent_all_reflexions[agent_id].append(reflexion) #To store all reflexions there have been
        
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

        #Summary is made from last summary + new reflexions
        if summary_method == "incremental":
            agent_summaries = [
                asyncio.create_task(
                GameOf24Agent.generate_summary(
                    agent_reflexions[agent_id], 
                    state=states[agent_id], 
                    api=step_batcher, 
                    namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                    )
                )
                for agent_id in states
            ]

        #Summary is made from all reflexions
        elif summary_method == "all_previous":
            agent_summaries = [
                asyncio.create_task(
                GameOf24Agent.generate_summary(
                    reflexion=agent_all_reflexions[agent_id], 
                    state=states[agent_id], 
                    api=step_batcher, 
                    namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                    )
                )
                for agent_id in states
            ]
        summaries = await asyncio.gather(*agent_summaries)

        for agent_id, summary in zip(states.keys(), summaries):
            agent_reflexions[agent_id] = [summary] #Replaces reflexions with summary
        return agent_reflexions, agent_all_reflexions
    else:
        raise ValueError("Unknown reflexion type")


async def run_reflexion_gameof24(states: Dict[int, GameOf24State], agent_ids: List[int], typeOfReflexion: str, num_reflexions: int, k: int, summary_method="incremental") -> int:
    """
    Runs a complete Game of 24 with reflexions.
    Returns the total score (number of succesful agents) of the agents.
    """
    puzzle = states[0].puzzle 
    agent_reflexions = {}
    agent_all_reflexions = {}
    num_steps = 4

    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
    total_score = 0

    #Reflect and go again i times
    for _ in range(num_reflexions):
        agent_reflexions, agent_all_reflexions = await make_reflexion(typeOfReflexion, k, states, agent_reflexions, agent_all_reflexions, summary_method)
        print("reflexions per agent", agent_reflexions)
        states, agent_ids, score = await solve_puzzle(num_steps, puzzle, agent_ids, agent_reflexions)
        total_score += score
    return total_score


async def run_step_wise_gameof24(states: Dict[int, GameOf24State], agent_ids: List[int], typeOfReflexion: str, num_reflexions: int, k: int, summary_method="incremental") -> int:
    """
    Runs a complete Game of 24 with step-wise reflexions.
    Returns the total score (number of succesful agents) of the agents.
    """
    #Started this approach where we basically just define the number of steps to be 1 and then run solve_puzzle() 4 times, however
    #Some changes needs to be made to solve_puzzle() for it to work, for example we need to be able to start from a specific step and 
    #We need to be able to pass the current states to the function.
    puzzle = states[0].puzzle
    agent_reflexions = {}
    agent_all_reflexions = {}
    num_steps = 1
    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
    total_score = 0

    for _ in range(num_reflexions):
        for i in range(4):
            agent_reflexions, agent_all_reflexions = await make_reflexion(typeOfReflexion, k, states, agent_reflexions, agent_all_reflexions, summary_method)
            print("reflexions per agent", agent_reflexions)
            states, agent_ids, score = await solve_puzzle(num_steps, puzzle, agent_ids, agent_reflexions)
            total_score += score
    return total_score


async def main():
    # Example of running an gameOf24 experiment with reflexion
    num_reflexions = 7
    k = 3
    num_agents = 4
    agent_ids = [i for i in range(num_agents)] #To keep track of active agents
    puzzles = load_test_puzzles()
    state = puzzles[0] #1, 1, 4, 6

    # await run_reflexion_gameof24(state, agent_ids, "summary", num_reflexions, k, "incremental")
    await run_step_wise_gameof24(state, agent_ids, "summary", num_reflexions, k, "incremental")
    

if __name__ == "__main__":
    asyncio.run(main())         
