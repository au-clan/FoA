# Imports
import asyncio
import os
import random
from secrets import token_bytes
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from diskcache import Cache
from deepdiff import DeepHash
sys.path.append(os.getcwd()) # Project root!!
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from src.agents.reflexionAgent import GameOf24Agent
from src.states.gameof24 import GameOf24State
from src.registry.registry import RegistryEntry
from utils import load_test_puzzles
from src.verifiers.RafaVerifiers import RafaVerifier
from src.verifiers.TextVerifier import TextVerifier
from src.validators.RafaValidator import RafaValidator
sys.stdout.reconfigure(encoding='utf-8')

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
    
# @dataclass(frozen=True)
# class RegistryEntry:
#     state: GameOf24State
#     verifiers: dict
#     reflexions: dict

LLMVERIFIER = False
IMPOSSIBLE_SCORE = 0.001
#LIKELY_SCORE     = 1.0  
#SURE_SCORE       = 20.0 

LOG_FILE = "mismatch_log.jsonl"
VERIFIER = RafaVerifier()

def set_LLMverifier(bool):
    global LLMVERIFIER
    LLMVERIFIER = bool
    

step_api_config = eval_api_config = {
    "max_tokens": 300,
    "temperature": 0.7,
    "top_p": 1,
    "request_timeout": 120,
    "top_k": 50
}

model = "llama-3.3-70b-versatile"
provider = "LazyKey"
#model = "gpt-4.1-nano-2025-04-14"
#provider = "OpenAI"
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

async def check_states(
    step_batcher: BatchingAPI, 
    type_of_reflexion,
    num_reflexions,
    step, 
    context,
    cache,
    agent_ids_to_check: List[int] = None
    ) -> None:
    """
    Checking whether the current state is valid and determines the likelihood of it succeding.
    """
    validator = RafaValidator()
    
    #Creating a list if using single_state with agent_ids_to_check otherwise creating list with all agents.
    if agent_ids_to_check is None:
        agent_ids = list(context.states.keys())
        # fully fresh run
        context.failed_agents.clear()
    else:
        print("agent_id to check")
        agent_ids = [agent_ids_to_check]
    
    valid_agents = []
    #Validate each agent state
    for agent_id in agent_ids:
        last_step = context.states[agent_id].steps[-2] if len(context.states[agent_id].steps) > 1 else context.states[agent_id].puzzle
        #print("last_step: ", last_step)
        #print("state in check_states: ", context.states[agent_id])
        validation = validator.validate_all(
            context.states[agent_id], 
            last_step
        )
        #Checking whether an agent's state is invalid or not
        if validation[1] == 0:
            context.agent_validations[agent_id] = validation
            context.failed_agents.append(agent_id) 
            context.agent_feedback[agent_id] = ("", "")
        else:
            valid_agents.append(agent_id)
            #Set validations to be empty
            context.agent_validations[agent_id] = ("", "")
    #Early stop if all agents contain invalid step
    if not valid_agents:
        return
    
    #If using the LLM verifier then the LLM determines whether it is still possible to reach 24.
    if LLMVERIFIER:  
        value_tasks = [
            asyncio.create_task(
            async_cache_verify( 
                context.states[agent_id],
                cache,
                agent_id,
                type_of_reflexion,
                num_reflexions,
                step_batcher,
                step
                )
            )
            for agent_id in valid_agents
        ]
        results = await asyncio.gather(*value_tasks)

        tokens_saved = 0
        price_saved = 0.0
        
        #Check if any values less than impossible score and mismatch detection
        for agent_id, (verification, (tokens, cost)) in zip(valid_agents, results):
            #print("\n\n")
            print("Value list: ", verification)
            context.agent_values[agent_id] = verification[0]
            iid_replies = verification[1]
            
            tokens_saved += tokens
            price_saved += cost

            #Need to run rafaverifier in order to do mismatching
            context.agent_feedback[agent_id] = verify(context.states[agent_id])            
                
            if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE:
                print("agent id: ", agent_id, " failed")
                if not agent_id in context.failed_agents: #Need this if for first steps before failed step
                    context.failed_agents.append(agent_id)
            else:
                if agent_id in context.failed_agents: 
                    context.failed_agents.remove(agent_id)

            mismatch_detecting(agent_id, context, step, iid_replies)
        return tokens_saved, price_saved
    else: 
        #Otherwise we use Rafa's deterministic verifier.
        for agent_id in valid_agents:
            feedback, reward = verify(context.states[agent_id])            
            if reward == 0 or "invalid" in feedback:
                print(f"agent {agent_id} failed")
                context.agent_feedback[agent_id] = (feedback, reward)
                if not agent_id in context.failed_agents:
                    context.failed_agents.append(agent_id)
            else: 
                if agent_id in context.failed_agents:
                    context.failed_agents.remove(agent_id) 
        return 0, 0
                    
async def async_cache_verify(state, cache, agent_id, type_of_reflexion, num_reflexions, step_batcher, step):
    # Create a proxy dict excluding 'randomness'
    state_to_hash = {
        'puzzle': state.puzzle,
        'current_state': state.current_state,
        'steps': state.steps
    }

    key = DeepHash(state_to_hash)[state_to_hash]
    entry = cache.get(key, RegistryEntry(state=state, verifiers={}, reflexions={}))
    verifier = VERIFIER
    print(f"agent_id has state: {state} and key: {key} in verification")
    
    if verifier.name in entry.verifiers:
        print("Getting verification from cache")
        verification = entry.verifiers[verifier.name]["verification"]
        #print("verification: ", verification)
        tokens_saved = entry.verifiers[verifier.name]["metadata"]["verification_tokens"]
        price_saved = entry.verifiers[verifier.name]["metadata"]["verification_cost"]
    else:
        print("Getting verification from LLM")
        cost = api.cost(tab_name="step_wise"+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=True)
        pre_cost = cost.get("total_tokens")
        cost = api.cost(tab_name="step_wise"+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=False)
        pre_price = cost.get("total_cost")
        verification = await GameOf24Agent.value( 
                            state,
                            step_batcher,
                            namespace=(0, f"Agent: {agent_id}", f"Step : {step}",)
                        )
        cost = api.cost(tab_name="step_wise"+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=True)
        post_cost = cost.get("total_tokens")
        cost = api.cost(tab_name="step_wise"+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=False)
        post_price = cost.get("total_cost")
        diff_cost = post_cost-pre_cost
        diff_price = post_price-pre_price
        print("diff_price: ", diff_price)
        #print("diff_cost: ", diff_cost)
        #print("verification: ", verification)
        metadata = {
        "model_used": model,
        "temperature": eval_api_config["temperature"],
        "max_tokens": eval_api_config["max_tokens"],
        "verification_tokens": diff_cost,
        "verification_cost": diff_price
        }
        entry.verifiers[verifier.name] = {"verification": verification, "metadata": metadata}
        cache.set(key, entry)
        tokens_saved = 0
        price_saved
    return verification, (tokens_saved, price_saved)
    

def mismatch_detecting(agent_id, context, step, iid_replies):
    feedback = context.agent_feedback[agent_id][0]
    reward = context.agent_feedback[agent_id][1]
    if reward == 1:
        if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE: #"Invalid" in feedback or
            #if "Invalid" in feedback:
                #log_mismatch(agent_id, step, context.states[agent_id], "False Negative", "Validation", context.agent_validations[agent_id], feedback)
            if context.agent_values[agent_id] <= IMPOSSIBLE_SCORE:
                log_mismatch(agent_id, step, context.states[agent_id], "False Negative", "Valuation", iid_replies, feedback)    

    # False Positive: validation or value says valid, but verifier reward = 0
    elif reward == 0:
        if context.agent_values[agent_id] > IMPOSSIBLE_SCORE: #"Valid" in feedback or
            # if "Valid" in feedback:
            #     log_mismatch(agent_id, step, context.states[agent_id], "False Positive", "Validation", context.agent_validations[agent_id], feedback)
            if context.agent_values[agent_id] > IMPOSSIBLE_SCORE:
                log_mismatch(agent_id, step, context.states[agent_id], "False Positive", "Valuation", iid_replies, feedback) 

def verify(state) -> Tuple[str, int]:
    verifier = VERIFIER
    return verifier.check_all(state)

@staticmethod
async def failed_agent_step(
    step: int,
    agent_id: int,
    context: AgentContext,
    type_of_reflexion: str,
    num_reflexions,
    k: int,
    agent_reflexions: Dict[int, List[str]],
    agent_all_reflexions: Dict[int, List[str]],
    log: Dict[str, Any],
    cache
    ):
    single_state = {agent_id: context.states[agent_id]}
    print("single_state: ", single_state)
    agent_reflexions, agent_all_reflexions, reflexion_tokens_saved, reflexion_price_saved = await make_reflexion(
        context.step_batcher, 
        "step_wise", 
        type_of_reflexion, 
        num_reflexions,
        k,
        single_state, 
        agent_reflexions, 
        agent_all_reflexions,
        context.agent_feedback[agent_id][0] ,
        context.agent_validations[agent_id][0],
        cache=cache
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
        return reflexion_tokens_saved, reflexion_price_saved

    verify_tokens_saved, verify_price_saved = await check_states(context.step_batcher, type_of_reflexion, num_reflexions, step, context, cache, agent_id)

    return reflexion_tokens_saved+verify_tokens_saved, reflexion_price_saved+verify_price_saved

async def solve_trial_wise(
        step_batcher: BatchingAPI,
        num_steps: int,
        iteration: int,
        puzzle_idx: int,
        puzzle: str, 
        agent_ids: List[int], 
        agent_reflexions: Dict[int, List[str]],
        log: Dict[str, Any]
    ) -> Tuple[Dict[int, GameOf24State], List[int], int]:
    """"
    Solves the puzzle either with or without reflexions.
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

    #print("agent_reflexions: ", agent_reflexions)
    #add last trials reflexion to log
    for agent_id in agent_ids:
        if iteration == 0:
            log_key = f"Agent {agent_id}"
        else:
            log_key = f"Agent {agent_id} - iteration {iteration}"
        log[log_key] = {
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
        # for agent_id in agent_ids:
        #     if f"Agent {agent_id}" not in log:
        #         log[f"Agent {agent_id}"] = {}
        for agent_id in agent_ids:
            if f"Agepuzzle_nt {agent_id} - iteration {iteration}" not in log[puzzle_idx]:
                log[puzzle_idx][f"Agent {agent_id} - iteration {iteration}"] = {}

        for agent_id, new_state in zip(states.keys(), new_states):
             # Log - Steps
            # log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{' -> '.join(new_state.steps)}"})
            step_description = new_state.steps[-1] if new_state.steps else "" #empty string should never happend
            log[puzzle_idx][f"Agent {agent_id} - iteration {iteration}"][f"Step {step}"] = {
                "Step": step_description
            }
            
            states[agent_id] = new_state
            print(f"Current step for agent {agent_id}: {new_state.steps[-1]} \n")
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
    return states, agent_ids, score, log


async def solve_step_wise(
        step_batcher: BatchingAPI,
        num_steps: int, 
        num_reflexions: int,
        k,
        puzzle: str, 
        agent_ids: List[int], 
        type_of_reflexion: str,
        log: Dict[str, Any],
        cache
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

    tokens_saved = 0
    price_saved = 0

    for agent_id in agent_ids:
        if f"Agent {agent_id}" not in log:
            log[f"Agent {agent_id}"] = {}
        
    for step in range(num_steps):
        print(f"Step {step} : Stepping")

        #Reset reflexions such that it does not use reflexions from earlier step
        for agent_id in agent_ids:
            agent_reflexions[agent_id] = []

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
            print(f"Current step for agent {agent_id}: {step_description}")
            
            context.states[agent_id] = new_state

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
    
        check_tokens_saved, check_price_saved = await check_states(step_batcher, type_of_reflexion, num_reflexions, step, context, cache)

        tokens_saved += check_tokens_saved
        price_saved += check_price_saved

        while context.failed_agents:
            for agent_id in context.failed_agents.copy():
                if agent_num_reflexions[agent_id] >= num_reflexions:
                    context.failed_agents.remove(agent_id)

            print("context failed agents: ", context.failed_agents)
            failed_agent_tasks = [
                asyncio.create_task(
                    failed_agent_step(
                        step=step,
                        agent_id=agent_id,
                        context=context,
                        type_of_reflexion=type_of_reflexion,
                        num_reflexions=num_reflexions,
                        k=k,
                        agent_reflexions=agent_reflexions,
                        agent_all_reflexions=agent_all_reflexions,
                        log=log,
                        cache=cache,
                    )
                )
                for agent_id in context.failed_agents.copy()
            ]

            if not failed_agent_tasks:
                print("breaks there are no failed agents with reflexions left")
                break

            for agent_id in context.failed_agents.copy():
                agent_num_reflexions[agent_id] += 1
                num_used_reflexions += 1

            # Wait for all tasks to complete
            results = await asyncio.gather(*failed_agent_tasks)

            for failed_tokens_saved, failed_price_saved in (results):
                tokens_saved += failed_tokens_saved
                price_saved += failed_price_saved

    with open("step_log.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
    return context.total_score, num_used_reflexions, tokens_saved, price_saved
   
async def async_cache_reflexion(states, agent_id, cache, time_of_reflexion, type_of_reflexion, num_reflexions, step_batcher, agent_feedback, agent_validation):
    state = states[agent_id]
    # Create a proxy dict excluding 'randomness'
    state_to_hash = {
        'puzzle': state.puzzle,
        'current_state': state.current_state,
        'steps': state.steps
    }

    key = DeepHash(state_to_hash)[state_to_hash]
    print(f"agent_id {agent_id} has state: {state} and key: {key} in make_reflexion")
    entry = cache.get(key, RegistryEntry(state=state, verifiers={}, reflexions={}))

    #If the reflection exists
    if time_of_reflexion in entry.reflexions:  
        reflexion = entry.reflexions[time_of_reflexion]["reflexion"]
        print("Getting reflexion from cache")
        tokens_saved = entry.reflexions[time_of_reflexion]["metadata"]["reflexion_tokens"]
        price_saved = entry.reflexions[time_of_reflexion]["metadata"]["reflexion_cost"]
    else:
        print("Getting reflexion from LLM")
        cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=True)
        pre_cost = cost.get("total_tokens")
        cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=False)
        pre_price = cost.get("total_cost")
        reflexion = await GameOf24Agent.generate_reflexion(
            time_of_reflexion,
            puzzle=states[agent_id].puzzle, 
            steps=states[agent_id].steps,
            state=states[agent_id],
            api=step_batcher, 
            namespace=(0, f"Agent: {int(agent_id)}"), 
            agent_feedback="" if LLMVERIFIER else agent_feedback,
            agent_validation= agent_validation
        )
        cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=True)
        post_cost = cost.get("total_tokens")
        cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+state.puzzle, report_tokens=False)
        post_price = cost.get("total_cost")
        
        diff_cost = post_cost - pre_cost
        diff_price = post_price - pre_price
        metadata = {
            "model_used": model,
            "temperature": eval_api_config["temperature"],
            "max_tokens": eval_api_config["max_tokens"],
            "reflexion_tokens": diff_cost,
            "reflexion_cost": diff_price
        }

        entry.reflexions[time_of_reflexion] = {"reflexion": reflexion, "metadata": metadata}
        cache.set(key, entry)
        tokens_saved = 0
        price_saved = 0
    return reflexion, (tokens_saved, price_saved)


async def make_reflexion(
        step_batcher: BatchingAPI,
        time_of_reflexion: str,
        type_of_reflexion: str,
        num_reflexions,
        k: int, 
        states: Dict[int, GameOf24State], 
        agent_reflexions: Dict[int, List[str]], 
        agent_all_reflexions: Dict[int, List[str]],
        agent_feedback: str = "",
        agent_validations = "",
        cache=None
    ) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Generates a reflexion for each agent based on their current state and the chosen type of reflexion.
    """
    cache_reflexions_tasks = [
        asyncio.create_task(
            async_cache_reflexion(
                states, agent_id, cache, time_of_reflexion, type_of_reflexion, num_reflexions, step_batcher, agent_feedback, agent_validations
            )
        )
        for agent_id in states
    ]
    results = await asyncio.gather(*cache_reflexions_tasks)
    #print("new_reflexions: ", new_reflexions)

    tokens_saved = 0
    price_saved = 0.0
        
    for agent_id, (reflexion, (tokens, price_saved)) in zip(states.keys(), results):
        agent_reflexions[agent_id].append(reflexion)
        agent_all_reflexions[agent_id].append(reflexion) #To store all reflexions there have been

        tokens_saved += tokens
        price_saved += price_saved
        
    if type_of_reflexion == "list":
        return agent_reflexions, agent_all_reflexions, tokens_saved, price_saved

    elif type_of_reflexion == "k_most_recent":
        for agent_id in agent_reflexions:
            agent_reflexions[agent_id] = agent_reflexions[agent_id][-k:]
        return agent_reflexions, agent_all_reflexions, tokens_saved, price_saved
        
    elif type_of_reflexion == "summary_incremental":
        for agent_id in states:
            print("before summarization for agent_id ", agent_id, "agent_reflexions: ", agent_reflexions[agent_id])
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
            print("after summarization for agent_id ", agent_id, "summary: ", summary)
        return agent_reflexions, agent_all_reflexions, tokens_saved, price_saved

        #Summary is made from all reflexions
    elif type_of_reflexion == "summary_all_previous":
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
        return agent_reflexions, agent_all_reflexions, tokens_saved, price_saved
    else:
        raise ValueError("Unknown reflexion type")


async def run_reflexion_gameof24(
        time_of_reflexion: str, 
        type_of_reflexion: str, 
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
        tab=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+puzzle
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
    log[puzzle_idx] = {"puzzle": puzzle, "type of reflexion": type_of_reflexion, "number of reflexions": num_reflexions}
    log[puzzle_idx].update({f"Agent {i}": {} for i in range(num_agents)})
    #Log initial solve
    print("state is : ", states[agent_id])
    if time_of_reflexion == "trial_wise":
        for step in range(num_steps):
            #for agent_id in range(num_agents):
            log[puzzle_idx][f"Agent {agent_id}"].update({f"Step {step}": {}})

            #for agent_id in range(num_agents):
            log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{states[agent_id].steps[step]}"})

    if model == "gpt-4.1-nano-2025-04-14":
        model_in_use = "openai"
    elif model == "llama-3.3-70b-versatile":
        model_in_use = "llama"
    cache = Cache(f'caches/registry - {model_in_use}')
    #print("states here: ", states)

    #Making states for all agent_ids, because pickle only has one
    # for agent_id in agent_ids:
    #     states[agent_id] = GameOf24State(
    #         puzzle=puzzle, 
    #         current_state=puzzle, 
    #         steps=[], 
    #         randomness=random.randint(0,1000)
    #         )


    total_tokens_saved = 0
    total_price_saved = 0.0
    # states, agent_ids, score = await solve_trial_wise(step_batcher, num_steps, puzzle_idx, puzzle, agent_ids, agent_reflexions, verifier)
    if time_of_reflexion == "trial_wise":
        #Reflect and go again i times
        for iteration in range(num_reflexions):
            agent_reflexions, agent_all_reflexions, tokens_saved, price_saved = await make_reflexion(step_batcher, time_of_reflexion, type_of_reflexion, num_reflexions, k, states, agent_reflexions, agent_all_reflexions, cache=cache)
            #print("tokens_saved: ", tokens_saved, "price_saved: ", price_saved)
            
            #print("reflexions per agent", agent_reflexions)
            states, agent_ids, score, log = await solve_trial_wise(step_batcher, num_steps, iteration, puzzle_idx, puzzle, agent_ids, agent_reflexions, log)
            total_score += score
            total_tokens_saved += tokens_saved
            total_price_saved += price_saved
        num_used_reflexions = sum(len(reflexions) for reflexions in agent_all_reflexions.values())
        #print(cache)
        with open("trial_log_test.jsonl", "a") as f:
            f.write(json.dumps(log, indent=4) + "\n")
    elif time_of_reflexion == "step_wise": #step_wise
        total_score, num_used_reflexions, total_tokens_saved, total_price_saved = await solve_step_wise(step_batcher, num_steps, num_reflexions, k, puzzle, agent_ids, type_of_reflexion, log, cache)
    else:
        print("Wrong time of reflexion")
    print("tokens_saved: ", total_tokens_saved)
    print("price_saved: ", total_price_saved)
    cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+puzzle, report_tokens=True)
    tokens_used = cost.get("total_tokens")
    print("tokens_used: ", tokens_used)
    cost = api.cost(tab_name=time_of_reflexion+type_of_reflexion+str(LLMVERIFIER)+str(num_reflexions)+puzzle, report_tokens=False)
    price_used = cost.get("total_cost")
    print("price_used: ", price_used)

    #For loop counting how many reflexions have been done in total 
    
    return total_score, tokens_used, tokens_saved, price_used, price_saved, num_used_reflexions


async def main():
    # Solve
    # Do reflexiongame
    # Example of running an gameOf24 experiment with reflexion
    num_reflexions = 4
    k = 2
    num_agents = 1
    puzzles = load_test_puzzles()
    state = puzzles[2] #6, 6, 6, 6

    for i in range(num_agents):
        state[i] = state[0]
    
    puzzle_idx = 0

    # await run_reflexion_gameof24(state, agent_ids, "summary", num_reflexions, k, "incremental")
    set_LLMverifier(True)
    total_score, tokens_used, tokens_saved, price_used, price_saved, num_used_reflexions = await run_reflexion_gameof24("trial_wise", "list", puzzle_idx, state, num_agents, num_reflexions, k) 
    print("total_score: ", total_score, "tokens_used: ", tokens_used, "tokens_saved: ", tokens_saved, "price_used: ", price_used, "price_saved: ", price_saved, "num_used_reflexions: ", num_used_reflexions)

    # total_score, tokens_used, num_used_reflexions = await run_reflexion_gameof24("trial_wise", "list", state, num_agents, num_reflexions, k, verifier) 
    # print("total_score: ", total_score, "tokens_used: ", tokens_used, "num_used_reflexions: ", num_used_reflexions)

if __name__ == "__main__":
    asyncio.run(main())       
    # state = GameOf24State("1 1 4 6", "test", "['test']", 2)
    # print(state.hash())
    # print(state.hash())
    cache = Cache('caches/registry')
    # cache.clear()
    # keys = []
    # for key in cache:
    #     keys.append(key)
    # print("number of keys in cache is ", len(keys))
    # for key in keys:
    #     print(key)
    #     print(cache.get(key))
        
    