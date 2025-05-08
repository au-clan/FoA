# Imports
import asyncio
import os
import random
from secrets import token_bytes
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
sys.path.append(os.getcwd()) # Project root!!
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from src.agents.reflexionAgent import GameOf24Agent
from src.states.gameof24 import GameOf24State
from utils import load_test_puzzles
from src.rafaverifiers import RafaVerifier

@dataclass
class AgentContext:
    states: Dict[int, GameOf24State]
    previous_states: Dict[int, GameOf24State]
    finished_states: Dict[int, GameOf24State]
    agent_ids: List[int]
    agent_feedback: Dict[int, str]
    agent_validations: Dict[int, str]
    agent_values: Dict[int, float]
    failed_agents: List[int]
    step_batcher: BatchingAPI
    total_score: int

LLMVERIFIER = False
IMPOSSIBLE_SCORE = 2.001 #TODO: Tag stilling til hvornår vi deemer noget impossible etc
#LIKELY_SCORE     = 1.0  
#SURE_SCORE       = 20.0 

RAFAVERIFIER = RafaVerifier()

def set_LLMverifier(bool):
    global LLMVERIFIER
    LLMVERIFIER = bool
    

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

LOG_FILE = "mismatch_log.jsonl"

def log_mismatch(agent_id, step, state, mismatch_type, source, message, RAFAVERIFIER_feedback):
    log_entry = {
        "agent_id": agent_id,
        "step": step,
        "state_steps": state.steps,
        "mismatch_type": mismatch_type,
        "source": source,
        "message": message,
        "verifier_feedback": RAFAVERIFIER_feedback
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

async def check_states(step_batcher: BatchingAPI, step, context) -> Tuple[str, int]:
    """
    Checking whether the current state is valid and determines the likelihood of it succeding.
    """
    value_tasks = [
        asyncio.create_task(
        GameOf24Agent.value(
            context.states[agent_id].puzzle, 
            context.states[agent_id].steps, 
            context.states[agent_id],
            step_batcher,
            namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
            )
        )
        for agent_id in context.states      
    ]
    values = await asyncio.gather(*value_tasks)
    
    for agent_id, value in zip(context.states.keys(), values):
        
        context.agent_values[agent_id] = value

        #Check what agents fails and append the agent id's to a list
        if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE:
            #print("check for invalid: ", "Invalid" in agent_validations[agent_id])
            #print("check for impossible: ", agent_values[agent_id] == IMPOSSIBLE_SCORE)
            print("agent id: ", agent_id, " failed")
            context.failed_agents.append(agent_id)

        if agent_id not in context.failed_agents:
            single_validation = await GameOf24Agent.validate(
                context.states[agent_id].puzzle, 
                context.states[agent_id].steps, 
                context.states[agent_id],
                step_batcher,
                namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                )
            context.agent_validations[agent_id] = single_validation
            if "Invalid" in single_validation:
                context.failed_agents.append(agent_id)

        mismatch_detecting(agent_id, context, step)



def mismatch_detecting(agent_id, context, step):
    feedback = context.agent_feedback[agent_id][0]
    reward = context.agent_feedback[agent_id][1]
    if reward == 1:
        if "Invalid" in context.agent_validations[agent_id] or context.agent_values[agent_id] <= IMPOSSIBLE_SCORE:
            if "Invalid" in context.agent_validations[agent_id]:
                log_mismatch(agent_id, step, context.states[agent_id], "False Negative", "Validation", context.agent_validations[agent_id], feedback)
            if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE:
                log_mismatch(agent_id, step, context.states[agent_id], "False Negative", "Valuation", context.agent_values[agent_id], feedback)    

    # False Positive: validation or value says valid, but verifier reward = 0
    elif reward == 0:
        if "Valid" in context.agent_validations[agent_id] or context.agent_values[agent_id] > IMPOSSIBLE_SCORE:
            if "Valid" in context.agent_validations[agent_id]:
                log_mismatch(agent_id, step, context.states[agent_id], "False Positive", "Validation", context.agent_validations[agent_id], feedback)
            if context.agent_values[agent_id] > IMPOSSIBLE_SCORE:
                log_mismatch(agent_id, step, context.states[agent_id], "False Positive", "Valuation", context.agent_values[agent_id], feedback) 

def verify(state, last_step) -> Tuple[str, int]:
    return RAFAVERIFIER.check_all(state, last_step)

@staticmethod
async def failed_agent_step(
    step: int,
    agent_id: int,
    context: AgentContext,
    reflexion_type: str,
    k: int,
    agent_reflexions: Dict[int, List[str]],
    agent_all_reflexions: Dict[int, List[str]],
    log: Dict[str, Any]
    ):
    single_state = {agent_id: context.states[agent_id]}
    print("single_state: ", single_state)
    agent_reflexions, agent_all_reflexions = await make_reflexion(
        context.step_batcher, "step_wise", reflexion_type, k,
        single_state, agent_reflexions, agent_all_reflexions,
        context.agent_feedback[agent_id][0] if not LLMVERIFIER else ""
    )
    # print("agent_id after reflexion: ", agent_id)
    #print("agent reflexions in step wise: ", agent_reflexions[agent_id])
    agent_task = asyncio.create_task(
        GameOf24Agent.step(
            context.previous_states[agent_id],
            context.step_batcher,
            namespace=(0, f"Agent: {agent_id}", f"Step : {step}"),
            reflexion=agent_reflexions[agent_id]
        )
    )
    reattempt_state = await asyncio.gather(agent_task) #Fake async, only for one state                    
    context.states[agent_id] = reattempt_state[0]
    log_entry = {
    "Step": context.states[agent_id].steps[-1] if context.states[agent_id].steps else "",
    "Reflexion": agent_reflexions[agent_id][-1] if agent_reflexions[agent_id] else "",
}
    # Ensure the dictionary exists
    if f"Step {step}" not in log[f"Agent {agent_id}"]:
        log[f"Agent {agent_id}"][f"Step {step}"] = log_entry
    else:
        # If already exists, you can append or overwrite depending on needs
        log[f"Agent {agent_id}"][f"Step {step}"].update(log_entry)

    print(f"Current step for agent {agent_id}: {context.states[agent_id].steps[-1]} \n")
        
    # Evaluate whether a puzzle has been solved, 
    if GameOf24Agent.verify(context.states[agent_id]) == {"r": 1}:
        print(f"Puzzle finished by agent {agent_id}: {context.states[agent_id].puzzle}")
        context.finished_states[agent_id] = context.states.pop(agent_id)
        context.agent_ids.remove(agent_id)
        context.failed_agents.remove(agent_id)
        context.total_score += 1
        return context.total_score

    #Deterministic verifier
    feedback, reward = verify(
        context.states[agent_id],
        context.states[agent_id].steps[-2] if len(context.states[agent_id].steps) > 1 else context.states[agent_id].puzzle
    )
    context.agent_feedback[agent_id] = (feedback, reward)
    
    if LLMVERIFIER:
        #Need to validate the new state
        #single_validation, single_value = await check_states(step_batcher, states, step)
        #TODO: Change check_states() to be able to accomodate single state / step
        single_value = await GameOf24Agent.value(
            context.states[agent_id].puzzle,
            context.states[agent_id].steps,
            context.states[agent_id],
            context.step_batcher,
            namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
        )
        context.agent_values[agent_id] = single_value

        if context.agent_values[agent_id] > IMPOSSIBLE_SCORE:
            single_validation = await GameOf24Agent.validate(
                context.states[agent_id].puzzle,
                context.states[agent_id].steps,
                context.states[agent_id],
                context.step_batcher,
                namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
            )
            context.agent_validations[agent_id] = single_validation

        #check if it fails or succeeds
        if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE or "Invalid" in context.agent_validations[agent_id]:
            print(f"agent {agent_id} failed again")
        else:
            context.failed_agents.remove(agent_id)

        mismatch_detecting(agent_id, context, step)
    else: 
        if reward > 0:
            context.failed_agents.remove(agent_id) 
        else:
            print(f"agent {agent_id} failed again")

async def solve_trial_wise(
        step_batcher: BatchingAPI,
        num_steps: int, 
        puzzle_idx: int,
        puzzle: str, 
        agent_ids: List[int], 
        agent_reflexions: Dict[int, List[str]],
        log: Dict[str, Any]
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

    #print("agent_reflexions: ", agent_reflexions)
    #add last trials reflexion to log
    log[f"Agent {agent_id}"] = {
            "Reflexion": agent_reflexions[agent_id][-1] if agent_reflexions[agent_id] else "",
        }
    #Stepping
    for step in range(num_steps):
        print(f"Step {step} : Stepping")
        
        # # Log - Set up log of each agent for current step
        # for agent_id in range(len(agent_ids)):
        #     log[puzzle_idx][f"Agent {agent_id}"].update({f"Step {step}": {}})
        
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
        
        #Ensures that the agent still exists in log
        for agent_id in agent_ids:
            if f"Agent {agent_id}" not in log:
                log[f"Agent {agent_id}"] = {}

        for agent_id, new_state in zip(states.keys(), new_states):
             # Log - Steps
            # log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{' -> '.join(new_state.steps)}"})
            step_description = new_state.steps[-1] if new_state.steps else "" #empty string should never happend
            log[f"Agent {agent_id}"][f"Step {step}"] = {
                "Step": step_description
            }
            
            states[agent_id] = new_state
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")
            if len(registry[agent_id]) == 0:
                last_step = new_state.puzzle
                registry[agent_id].append((new_state.steps[-1], verify(new_state, last_step)))
            else:
                registry[agent_id].append((new_state.steps[-1], verify(new_state, registry[agent_id][-1][0])))
        #print(registry)
        # Evaluate whether a puzzle has been solved, 
        for agent_id in list(states.keys()):
            if GameOf24Agent.verify(states[agent_id]) == {"r": 1}:
                print(f"Puzzle finished by agent {agent_id}: {states[agent_id].puzzle}")
                finished_states[agent_id] = states.pop(agent_id)
                agent_ids.remove(agent_id)
                score +=1
        
        # After each step, the api should be empty
        assert len(step_batcher.futures) == 0, f"API futures should be empty, but are {len(step_batcher.futures)}"
        # If all puzzles have been solved, break
        if not states:
            break
    with open("trial_log.jsonl", "a") as f:
        f.write(json.dumps(log, indent=4) + "\n")
    return states, agent_ids, score#, log


async def solve_step_wise(
        step_batcher: BatchingAPI,
        num_steps: int, 
        num_reflexions: int,
        k,
        puzzle: str, 
        agent_ids: List[int], 
        reflexion_type: str,
        log: Dict[str, Any]
    ) -> Tuple[Dict[int, GameOf24State], List[int], int]:

    states = {} 
    finished_states = {}
    agent_reflexions = {}
    agent_all_reflexions = {}
    agent_num_reflexions = {}
    agent_feedback = {}
    num_used_reflexions = 0
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
        agent_feedback[agent_id] = ""


    context = AgentContext(
        states=states,
        previous_states={},
        finished_states=finished_states,
        agent_ids=agent_ids,
        agent_feedback=agent_feedback,
        agent_validations={},
        agent_values={},
        failed_agents=[],
        step_batcher=step_batcher,
        total_score=0
    )

    for agent_id in agent_ids:
        if f"Agent {agent_id}" not in log:
            log[f"Agent {agent_id}"] = {}
        
    for step in range(num_steps):
        print(f"Step {step} : Stepping")

        #Save previous valid step before stepping
        context.previous_states = states.copy()
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
            #logging
            step_description = new_state.steps[-1] if new_state.steps else ""
            log[f"Agent {agent_id}"][f"Step {step}"] = {
                "Step": step_description,
                "Reflexion": agent_reflexions[agent_id][-1] if agent_reflexions[agent_id] else "",
            }

            context.states[agent_id] = new_state
            #print("previous_states:", previous_states[agent_id])
            if len(context.previous_states[agent_id].steps) == 0:
                last_step = states[agent_id].puzzle
            else:
                last_step = context.previous_states[agent_id].steps[-1]
            feedback, reward = verify(new_state, last_step)
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")
            context.agent_feedback[agent_id] = (feedback, reward)
        #print(states)
        # Evaluate whether a puzzle has been solved, 
        for agent_id in list(states.keys()):
            if GameOf24Agent.verify(states[agent_id]) == {"r": 1}:
                print(f"Puzzle finished by agent {agent_id}: {states[agent_id].puzzle}")
                context.finished_states[agent_id] = states.pop(agent_id)
                context.agent_ids.remove(agent_id)
                context.total_score +=1

        # If all puzzles have been solved, break
        if not states:
            break
    
        if LLMVERIFIER:
            await check_states(step_batcher, step, context)   
        else:
            for agent_id in context.agent_ids:
                if context.agent_feedback[agent_id][1] == 0:
                    context.failed_agents.append(agent_id)   

        while context.failed_agents:
            print("context failed agents: ", context.failed_agents)
            failed_agent_tasks = [
                asyncio.create_task(
                    failed_agent_step(
                        step=step,
                        agent_id=agent_id,
                        context=context,
                        reflexion_type=reflexion_type,
                        k=k,
                        agent_reflexions=agent_reflexions,
                        agent_all_reflexions=agent_all_reflexions,
                        log=log,
                    )
                )
                for agent_id in context.failed_agents.copy()
                if agent_num_reflexions[agent_id] < num_reflexions
            ]

            if not failed_agent_tasks:
                print("breaks there are no failed agents with reflexions left")
                break

            for agent_id in context.failed_agents.copy():
                agent_num_reflexions[agent_id] += 1
                num_used_reflexions += 1

            # Wait for all tasks to complete
            await asyncio.gather(*failed_agent_tasks)

    with open("trial_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    return context.total_score, num_used_reflexions
   

async def make_reflexion(
        step_batcher: BatchingAPI,
        time_of_reflexion: str,
        reflexion_type: str,
        k: int, 
        states: Dict[int, GameOf24State], 
        agent_reflexions: Dict[int, List[str]], 
        agent_all_reflexions: Dict[int, List[str]],
        agent_feedback: str = ""
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Generates a reflection for each agent based on their current state and the chosen type of reflection.
    """

    print("states: ", states)
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(
                time_of_reflexion,
                puzzle=states[agent_id].puzzle, 
                steps=states[agent_id].steps, 
                state=states[agent_id],
                api=step_batcher, 
                namespace=(0, f"Agent: {int(agent_id)}"), 
                agent_feedback=agent_feedback
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
            print("before summarization for agent_id: ", agent_id, "agent_reflexions: ", agent_reflexions[agent_id])
        agent_summaries = []

        #Summary is made from last summary + new reflexions
        agent_summaries = [
            asyncio.create_task(
            GameOf24Agent.generate_summary(
                agent_reflexions[agent_id], 
                state=states[agent_id], 
                api=step_batcher, 
                namespace=(0, f"Agent: {agent_id}")
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
        for agent_id in states:
            print("before summarization for agent_id: ", agent_id, "agent_reflexions: ", agent_reflexions[agent_id])
        agent_summaries = [
            asyncio.create_task(
            GameOf24Agent.generate_summary(
                reflexion=agent_all_reflexions[agent_id], 
                state=states[agent_id], 
                api=step_batcher, 
                namespace=(0, f"Agent: {agent_id}")
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
        puzzle_idx: int,
        states: Dict[int, GameOf24State], 
        num_agents: int, 
        num_reflexions: int, 
        k: int
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
    # Set up log
    log = {}
    log[puzzle_idx] = {"puzzle": puzzle}
    log[puzzle_idx].update({f"Agent {i}": {} for i in range(num_agents)})
    #Log initial solve
    print("num agents: ", num_agents)
    print("state is : ", states[agent_id])
    if time_of_reflexion == "trial_wise":
        for step in range(num_steps):
            for agent_id in range(num_agents):
                log[puzzle_idx][f"Agent {agent_id}"].update({f"Step {step}": {}})

            #for agent_id in range(num_agents):
                log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{states[agent_id].steps[step]}"})
                print("Logged step: ", states[agent_id].steps[step])

    #print("states here: ", states)

    #Making states for all agent_ids, because pickle only has one
    # for agent_id in agent_ids:
    #     states[agent_id] = GameOf24State(
    #         puzzle=puzzle, 
    #         current_state=puzzle, 
    #         steps=[], 
    #         randomness=random.randint(0,1000)
    #         )

    # states, agent_ids, score = await solve_trial_wise(step_batcher, num_steps, puzzle_idx, puzzle, agent_ids, agent_reflexions, verifier)
    if time_of_reflexion == "trial_wise":
        #Reflect and go again i times
        for _ in range(num_reflexions):
            agent_reflexions, agent_all_reflexions = await make_reflexion(step_batcher, time_of_reflexion, reflexion_type, k, states, agent_reflexions, agent_all_reflexions)
            print("reflexions per agent", agent_reflexions)
            states, agent_ids, score = await solve_trial_wise(step_batcher, num_steps, puzzle_idx, puzzle, agent_ids, agent_reflexions, log)
            total_score += score
        num_used_reflexions = sum(len(reflexions) for reflexions in agent_all_reflexions.values())
    else: #step_wise
        total_score, num_used_reflexions = await solve_step_wise(step_batcher, num_steps, num_reflexions, k, puzzle, agent_ids, reflexion_type, log)
    cost = api.cost(tab_name=reflexion_type+str(num_reflexions)+puzzle, report_tokens=True)
    token_cost = cost.get("total_tokens")

    #For loop counting how many reflexions have been done in total 
    
    return total_score, token_cost, num_used_reflexions


async def main():
    # Solve
    # Do reflexiongame
    # Example of running an gameOf24 experiment with reflexion
    num_reflexions = 2
    k = 2
    num_agents = 2
    puzzles = load_test_puzzles()
    state = puzzles[0] #1, 1, 4, 6

    for i in range(num_agents):
        state[i] = state[0]
    
    puzzle_idx = 0

    # await run_reflexion_gameof24(state, agent_ids, "summary", num_reflexions, k, "incremental")
    set_LLMverifier(False)
    total_score, token_cost, num_used_reflexions = await run_reflexion_gameof24("trial_wise", "list", puzzle_idx, state, num_agents, num_reflexions, k) 
    print("total_score: ", total_score, "token_cost: ", token_cost, "num_used_reflexions: ", num_used_reflexions)

    # total_score, token_cost, num_used_reflexions = await run_reflexion_gameof24("trial_wise", "list", state, num_agents, num_reflexions, k, verifier) 
    # print("total_score: ", total_score, "token_cost: ", token_cost, "num_used_reflexions: ", num_used_reflexions)

if __name__ == "__main__":
    asyncio.run(main())         
