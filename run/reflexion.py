# Imports
import asyncio
import os
import random
from secrets import token_bytes
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
sys.path.append(os.getcwd()) # Project root!!
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from src.agents.reflexionAgent import GameOf24Agent
from src.states.gameof24 import GameOf24State
from utils import load_test_puzzles
from src.rafaverifiers import RafaVerifier


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

async def check_states(step_batcher, states, step):
    """
    Checking whether the current state is valid and determines the likelihood of it succeding.
    """
    validate_tasks = [
            asyncio.create_task(
            GameOf24Agent.validate(
                states[agent_id].puzzle, 
                states[agent_id].steps, 
                states[agent_id],
                step_batcher,
                namespace=(0, f"Agent: {agent_id}", f"Step : {step}") #TODO: fix namespace stuff
                )
            )
            for agent_id in states
        ]
    validations = await asyncio.gather(*validate_tasks)

    value_tasks = [
        asyncio.create_task(
        GameOf24Agent.value(
            states[agent_id].puzzle, 
            states[agent_id].steps, 
            states[agent_id],
            step_batcher,
            namespace=(0, f"Agent: {agent_id}", f"Step : {step}") #TODO: fix namespace stuff
            )
        )
        for agent_id in states      
    ]
    values = await asyncio.gather(*value_tasks)

    return validations, values

def verify(state, last_step, Verifier) -> Tuple[str, int]:
    return Verifier.check_all(state, last_step)

async def solve_trial_wise(
        step_batcher: BatchingAPI,
        num_steps: int, 
        puzzle: str, 
        agent_ids: List[int], 
        agent_reflexions: Dict[int, List[str]],
        verifier
    ) -> Tuple[Dict[int, GameOf24State], List[int], int]:
    """"
    Solves the puzzle either with or without reflections.
    Returns the updated states, remaining agent_ids, and the accumulated score.
    """
    score = 0
    states =  {}
    registry = {}
    #Create one state for each agent
    for agent_id in agent_ids:
        states[agent_id] = GameOf24State(
            puzzle=puzzle, 
            current_state=puzzle, 
            steps=[], 
            randomness=random.randint(0,1000)
            )
        registry[agent_id] = []
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
            if len(registry[agent_id]) == 0:
                last_step = new_state.puzzle
                registry[agent_id].append((new_state.steps[-1], verify(new_state, last_step, verifier)))
            else:
                registry[agent_id].append((new_state.steps[-1], verify(new_state, registry[agent_id][-1][0], verifier)))
        print(registry)
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


async def solve_step_wise(
        step_batcher: BatchingAPI,
        num_steps: int, 
        num_reflexions: int,
        k,
        puzzle: str, 
        agent_ids: List[int], 
        reflexion_type: str,
        verifier
    ) -> Tuple[Dict[int, GameOf24State], List[int], int]:

    total_score = 0
    states = {} 
    finished_states = {}
    agent_reflexions = {}
    agent_all_reflexions = {}
    agent_num_reflexions = {}

    #Create one state for each agent
    for agent_id in agent_ids:
        states[agent_id] = GameOf24State(
            puzzle=puzzle, 
            current_state=puzzle, 
            steps=[], 
            randomness=random.randint(0,1000)
            )

    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
        agent_num_reflexions[agent_id] = 0

    for step in range(num_steps):
        print(f"Step {step} : Stepping")
        agent_validations = {}
        agent_values = {}

        #Save previous valid step before stepping
        previous_states = states.copy()
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
            print("previous_states:", previous_states[agent_id])
            if len(previous_states[agent_id].steps) == 0:
                last_step = states[agent_id].puzzle
            else:
                last_step = previous_states[agent_id].steps[-1]
            feedback, reward = verify(new_state, last_step, verifier)
            print("feedback is: ", feedback)
            print("Reward is: ", reward)
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")
        print(states)
        # Evaluate whether a puzzle has been solved, 
        for agent_id in list(states.keys()):
            if GameOf24Agent.verify(states[agent_id]) == {"r": 1}:
                print(f"Puzzle finished by agent {agent_id}: {states[agent_id].puzzle}")
                finished_states[agent_id] = states.pop(agent_id)
                agent_ids.remove(agent_id)
                total_score +=1

        # If all puzzles have been solved, break
        if not states:
            break

        validations, values = await check_states(step_batcher, states, step)

        failed_agents = []
        for agent_id, validation, value in zip(states.keys(), validations, values):
            agent_validations[agent_id] = validation
            agent_values[agent_id] = value
            
            print("validation: ", validation)
            print("valuation: ", value)
            #Check what agents fails and append the agent id's to a list
            if "Invalid" in agent_validations[agent_id] or "Impossible" in agent_values[agent_id]:
                print("check for invalid: ", "Invalid" in agent_validations[agent_id])
                print("check for impossible: ","Impossible" in agent_values[agent_id])
                print("agent id: ", agent_id, " failed")
                failed_agents.append(agent_id)
            
        #Make it async
        while (failed_agents != []):
            for agent_id in failed_agents:
                print("agent_id in failed agent loop: ", agent_id)
                if agent_num_reflexions[agent_id] >= num_reflexions:
                    failed_agents.remove(agent_id)
                else:
                    single_state = {agent_id: states[agent_id]}
                    agent_num_reflexions[agent_id] += 1
                    agent_reflexions, agent_all_reflexions = await make_reflexion(step_batcher, "step_wise", reflexion_type, k, single_state, agent_reflexions, agent_all_reflexions)
                    # print("agent_id after reflexion: ", agent_id)
                    print("agent reflexions in step wise: ", agent_reflexions[agent_id])
                    agent_tasks = [
                        asyncio.create_task(
                        GameOf24Agent.step(
                            previous_states[agent_id], 
                            step_batcher, 
                            namespace=(0, f"Agent: {agent_id}", f"Step : {step}"), #TODO: fix namespace stuff
                            reflexion=agent_reflexions[agent_id]) 
                        )
                    ]
                    reattempt_state = await asyncio.gather(*agent_tasks) #Fake async, only for one state                    
                    states[agent_id] = reattempt_state[0]
                    print(f"Current step for agent {agent_id}: {states[agent_id].steps[-1]} \n")
                        
                    # Evaluate whether a puzzle has been solved, 
                    for new_agent_id in list(states.keys()):
                        if GameOf24Agent.verify(states[new_agent_id]) == {"r": 1}:
                            print(f"Puzzle finished by agent {new_agent_id}: {states[new_agent_id].puzzle}")
                            finished_states[new_agent_id] = states.pop(new_agent_id)
                            agent_ids.remove(new_agent_id)
                            total_score +=1
                    
                    #Need to validate the new state
                    #single_validation, single_value = await check_states(step_batcher, states, step)
                    validate_tasks = [
                        asyncio.create_task(
                            GameOf24Agent.validate(
                                states[agent_id].puzzle, 
                                states[agent_id].steps, 
                                states[agent_id],
                                step_batcher,
                                namespace=(0, f"Agent: {agent_id}", f"Step : {step}") #TODO: fix namespace stuff
                            )
                        )
                    ]
                    single_validation = await asyncio.gather(*validate_tasks)

                    value_tasks = [
                        asyncio.create_task(
                            GameOf24Agent.value(
                                states[agent_id].puzzle, 
                                states[agent_id].steps, 
                                states[agent_id],
                                step_batcher,
                                namespace=(0, f"Agent: {agent_id}", f"Step : {step}") #TODO: fix namespace stuff
                            )
                        )
                    ]
                    single_value = await asyncio.gather(*value_tasks)
                    agent_validations[agent_id] = single_validation[0]
                    agent_values[agent_id] = single_value[0]
                    print("validation for failed agent: ", agent_validations[agent_id])
                    print("valuations for failed agent: ", agent_values[agent_id])
                    #check if it fails or succeeds
                    if "Invalid" in agent_validations[agent_id] or "Impossible" in agent_values[agent_id]:
                        print("agent id: ", agent_id, " failed again")
                    else:
                        failed_agents.remove(agent_id)

    return total_score
   

async def make_reflexion(
        step_batcher: BatchingAPI,
        time_of_reflexion: str,
        reflexion_type: str,
        k: int, 
        states: Dict[int, GameOf24State], 
        agent_reflexions: Dict[int, List[str]], 
        agent_all_reflexions: Dict[int, List[str]]
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Generates a reflection for each agent based on their current state and the chosen type of reflection.
    """
    print("states: ", states)
    step = 3
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(
                time_of_reflexion,
                puzzle=states[agent_id].puzzle, 
                steps=states[agent_id].steps, 
                state=states[agent_id], 
                api=step_batcher, 
                namespace=(0, f"Agent: {int(agent_id)}", f"Step: {step}") #TODO: Change namespace to not have step
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
        
    elif reflexion_type == "summary_incremental":
        for agent_id in states:
            print("for agent_id: ", agent_id, "agent_reflexions: ", agent_reflexions[agent_id])
        agent_summaries = []

        #Summary is made from last summary + new reflexions
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
        summaries = await asyncio.gather(*agent_summaries)

        for agent_id, summary in zip(states.keys(), summaries):
            agent_reflexions[agent_id] = [summary] #Replaces reflexions with summary
        return agent_reflexions, agent_all_reflexions

        #Summary is made from all reflexions
    elif reflexion_type == "summary_all_previous":
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


async def run_reflexion_gameof24(
        time_of_reflexion: str, 
        reflexion_type: str, 
        states: Dict[int, GameOf24State], 
        num_agents: int, 
        num_reflexions: int, 
        k: int,
        verifier
    ) -> int:
    """
    Runs a complete Game of 24 with reflexions.
    Returns the total score (number of succesful agents) of the agents.
    """
    puzzle = states[0].puzzle
    step_batcher = BatchingAPI(
        api, 
        batch_size=1, 
        timeout=2, 
        model=models["step"]["model_name"], 
        tab=reflexion_type+str(num_reflexions)+puzzle
    )
    agent_reflexions = {}
    agent_all_reflexions = {}
    num_steps = 4
    agent_ids = [i for i in range(num_agents)] # Number of active agents

    for agent_id in agent_ids:
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []
    total_score = 0

    #print("states here: ", states)

    #Making states for all agent_ids, because pickle only has one
    # for agent_id in agent_ids:
    #     states[agent_id] = GameOf24State(
    #         puzzle=puzzle, 
    #         current_state=puzzle, 
    #         steps=[], 
    #         randomness=random.randint(0,1000)
    #         )

    # states, agent_ids, score = await solve_trial_wise(num_steps, puzzle, agent_ids, agent_reflexions)
    if time_of_reflexion == "trial_wise":
        #Reflect and go again i times
        for _ in range(num_reflexions):
            agent_reflexions, agent_all_reflexions = await make_reflexion(step_batcher, time_of_reflexion, reflexion_type, k, states, agent_reflexions, agent_all_reflexions)
            print("reflexions per agent", agent_reflexions)
            states, agent_ids, score = await solve_trial_wise(step_batcher, num_steps, puzzle, agent_ids, agent_reflexions, verifier)
            total_score += score
    else: #step_wise
        total_score = await solve_step_wise(step_batcher, num_steps, num_reflexions, k, puzzle, agent_ids, reflexion_type, verifier)
    cost = api.cost(tab_name=reflexion_type+str(num_reflexions)+puzzle, report_tokens=True)
    token_cost = cost.get("total_tokens")
    num_used_reflexions = sum(len(reflexions) for reflexions in agent_all_reflexions.values())
    
    #For loop counting how many reflexions have been done in total 
    
    return total_score, token_cost, num_used_reflexions


async def main():
    # Solve
    # Do reflexiongame
    # Example of running an gameOf24 experiment with reflexion
    num_reflexions = 2
    k = 3
    num_agents = 2
    puzzles = load_test_puzzles()
    state = puzzles[0] #1, 1, 4, 6
    verifier = RafaVerifier()

    # await run_reflexion_gameof24(state, agent_ids, "summary", num_reflexions, k, "incremental")
    total_score, token_cost, num_used_reflexions = await run_reflexion_gameof24("step_wise", "list", state, num_agents, num_reflexions, k, verifier) #this does not work atm
    print("total_score: ", total_score, "token_cost: ", token_cost, "num_used_reflexions: ", num_used_reflexions)

    # total_score, token_cost, num_used_reflexions = await run_reflexion_gameof24("trial_wise", "list", state, num_agents, num_reflexions, k, verifier) 
    # print("total_score: ", total_score, "token_cost: ", token_cost, "num_used_reflexions: ", num_used_reflexions)

if __name__ == "__main__":
    asyncio.run(main())         
