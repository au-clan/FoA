# Imports

from dataclasses import dataclass
from typing import List

import random

import asyncio
import re
import math
import random
import numpy as np
from sympy import simplify

random.seed(0)

from async_engine.batched_api import BatchingAPI
from async_engine.api import API

from src.prompts.adapt import gameof24 as llama_prompts
from utils import parse_suggestions, create_box


from lazykey import AsyncKeyHandler
from groq import AsyncGroq

# State class

@dataclass(frozen=True)
class GameOf24State:
    # game of 24 puzzle, for example 1 1 4 6
    puzzle: str

    # initialized to the same value as puzzle, but is updated as the game progresses
    current_state: str

    steps: List[str]

    #Randomness used for resampling (random seed)
    randomness: int

    def __hash__(self):
        return hash((self.puzzle, self.current_state, " -> ".join(self.steps)))
    
    def items(self):
        return self.puzzle, self.current_state, self.steps, self.randomness
    
    def duplicate(self, randomness=None):
        return GameOf24State(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            randomness=randomness if randomness is not None else self.randomness)


#Reflexion agent :O

class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api, namespace, reflexion: list)-> GameOf24State:
        """
        Given a state, returns the next state one.
        """

        # set up the prompt, based on the current state

        # ToT uses bfs_prompt to generate next steps but then uses
        # the cot_prompt to get the final expression. 
        # For example, input : 1 1 4 6
        # Step 0 : '1 - 1 = 0 (left: 0 4 6)'          BFS prompt
        # Step 1 : '0 + 4 = 4 (left: 4 6)'            BFS prompt
        # Step 2 : '4 * 6 = 24 (left: 24)'            BFS prompt
        # Step 3 : Answer : ((1 - 1) + 4) * 6 = 24    CoT prompt


        # set up the prompt, based on the current state
        current_state = state.current_state
        
        if current_state.strip() == "24":
            # CoT prompt
            steps = "\n".join(state.steps) + "\n"
            
            prompt = llama_prompts.cot_prompt.format(input=state.puzzle) + "Steps:\n" + steps + "Answer: "

            # Get the final expression
            suggestions = await api.buffered_request(prompt, key=hash(state), namespace=namespace)

            # State does not change, only the steps
            selected_suggestion = suggestions
            selected_state = state.current_state
            


        else:
            if len(reflexion) == 0:
                prompt = llama_prompts.bfs_prompt.format(input=current_state) 
            else:
                prompt = llama_prompts.bfs_reflexion_prompt.format(input=current_state, puzzle = "1 1 4 6", reflexion=reflexion[0]) 

            suggestions = await api.buffered_request(prompt, key=hash(state), namespace=namespace)

            # parse suggestions, based on the current state
            parsed_suggestions = parse_suggestions(suggestions)
            if parsed_suggestions == []:
                print(f"No suggestions were parsed from state: {state}")
                print(f"\nPrompt: {prompt}\nSuggestions: {suggestions}\nParsed suggestions: {' | '.join(parsed_suggestions)}\n")
                assert False, "No suggestions found."
            
            suggestions = parsed_suggestions
            
            random.seed(state.randomness)
            selected_suggestion = random.choice(suggestions)
            selected_state = GameOf24Agent.parse_next_state(selected_suggestion)

        # set up new state object
        next_state = GameOf24State(
            puzzle=state.puzzle,
            current_state=selected_state,
            steps=state.steps + [selected_suggestion],
            randomness=random.randint(0, 1000)
        )
        return next_state
    
    @staticmethod
    def parse_next_state(suggestion: str) -> str:
        return suggestion.split('left: ')[-1].split(')')[0]
    
    @staticmethod
    def verify(state: GameOf24State)-> dict:
            """
            Verifies the output of a given task
                1. Checks if the numbers used are the same as the ones provided.
                2. Checks if the operations performed result to 24.

            States 
                {"r": 0} : Not finished.
                {"r": 1} : Finished and correct.
                {"r": -1} : Finished and incorrect.
            """
            current_states = state.current_state.split(" ")
            if len(current_states) !=1 or len(state.steps)<=3:
                # More than one number left
                return {'r':0}
            elif current_states[0] != "24":
                # One number left and it is not 24
                return {'r':-1}
            else:
                # One number left and it is 24
                expression = state.steps[-1].lower().replace('answer: ', '').split('=')[0]
                numbers = re.findall(r'\d+', expression)
                problem_numbers = re.findall(r'\d+', state.puzzle)
                if sorted(numbers) != sorted(problem_numbers):
                    # Numbers used are not the same as the ones provided
                    return {'r': -1}
                try:
                    if simplify(expression) == 24:
                        return {'r': 1}
                    else:
                        # Operations performed do not result to 24
                        return {'r': -1}
                except Exception as e:
                    print(e)
                    return {'r': -1}

    @staticmethod
    def generate_reflexion(puzzle: str, steps, state: GameOf24State, api, namespace) -> str:
        prompt = llama_prompts.reflexion_prompt.format(puzzle=puzzle, steps=steps)
        reflexion = api.buffered_request(prompt, key=hash(state), namespace=namespace)
        return reflexion

    @staticmethod
    def generate_summary(reflexion, state: GameOf24State, api, namespace) -> str:
        prompt = llama_prompts.summary_prompt.format(reflexion=reflexion)
        reflexion = api.buffered_request(prompt, key=hash(state), namespace=namespace)
        return reflexion

# Attempting to solve the puzzle
async def solvePuzzle(agent_reflexions):
    #Create initial state/environment
    states =  {}
    for agent_id in range(num_agents):
        states[agent_id] = GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0,1000))
        agent_reflexions[agent_id] = []
        agent_all_reflexions[agent_id] = []

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

        # If all puzzles have been solved, break
        if not states:
            break
    return states

async def makeReflexion(reflexion_type, num_reflexions, k, states, agent_reflexions, summary_method):
    step = 3
    agent_tasks = [
        asyncio.create_task(
            GameOf24Agent.generate_reflexion(puzzle=puzzle, steps=states[agent_id].steps, state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {agent_id}", f"Step: {step}")
        )
    )
    for agent_id in states
    ] 

    new_reflexions = await asyncio.gather(*agent_tasks)

    for agent_id, reflexion in zip(states.keys(), new_reflexions):
        agent_reflexions[agent_id].append(reflexion)
        agent_all_reflexions[agent_id].append(reflexion)

    if reflexion_type == "list":
        print("reflexions per agent:", agent_reflexions)
        return agent_reflexions, agent_all_reflexions

    elif reflexion_type == "k most recent":
        for agent_id in agent_reflexions:
            agent_reflexions[agent_id] = agent_reflexions[agent_id][-k:]
        print("Reflexions per agent (k most recent):", agent_reflexions)
        return agent_reflexions, agent_all_reflexions

    elif reflexion_type == "summary":
        #Right now makes summary of earlier summary + new reflexions, 
        # if we want to change this we need to return reflexion and summary, pass summary to solvePuzzle, pass reflexion to makeReflexion
        agent_summaries = []
        if summary_method == "incremental":
            agent_summaries = [
                asyncio.create_task(
                GameOf24Agent.generate_summary(reflexion=agent_reflexions[agent_id][-1] if len(agent_reflexions[agent_id]) > 1 else agent_reflexions[agent_id], 
                state=states[agent_id], api=step_batcher, namespace=(0, f"Agent: {agent_id}", f"Step : {step}")
                    )
                )
                for agent_id in states
            ]
        elif summary_method == "all_previous":
            gent_summaries = [
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
        print("Summaries per agent", agent_reflexions)
        return agent_reflexions, agent_all_reflexions
    else:
        print("unknown type")
        return agent_reflexions, agent_all_reflexions

async def runReflexionGameOf24(typeOfReflexion, num_iterations, k, summary_method="incremental"):
    agent_reflexions = {}
    agent_all_reflexions = {} #for summaries based on all reflexions
    
    #Without reflexion first
    states = await solvePuzzle(agent_reflexions)
    print(states)
    #Reflect and go again i times
    for i in range(num_iterations):
        agent_reflexions, agent_all_reflexions = await makeReflexion(typeOfReflexion, i+1, k, states, agent_reflexions, summary_method)
        states = await solvePuzzle(agent_reflexions)

step_api_config = eval_api_config = {
    "max_tokens": 1000,
    "temperature": 0,
    "top_p": 1,
    "request_timeout": 120,
    "top_k": 50
}

model = "llama-3.3-70b-versatile"
provider = "Groq"
models = {
    "step": {"model_name":model, "provider":provider},
    "eval": {"model_name":model, "provider":provider},
}

api_keys = ["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", "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", "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"]
client = AsyncKeyHandler(api_keys, AsyncGroq)

api = API(eval_api_config, client=client, models=models.values(), resources=2, verbose=False)

puzzle = "1 1 4 6"
num_steps = 4
num_agents = 2


step_batcher = BatchingAPI(api, batch_size=1, timeout=2, model=models["step"]["model_name"], tab="step")

#Dictionary for reflexions for each agent
agent_reflexions = {}
agent_all_reflexions = {}


if __name__ == "__main__":
    asyncio.run(runReflexionGameOf24("list", 2, 2))
    asyncio.run(runReflexionGameOf24("k most recent", 2, 1))
    asyncio.run(runReflexionGameOf24("summary", 2, 2, "incremental"))
    asyncio.run(runReflexionGameOf24("summary", 2, 2, "all_previous"))
