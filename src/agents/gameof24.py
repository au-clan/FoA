import asyncio
import re
import math
import random
import numpy as np
from sympy import simplify

random.seed(0)

from src.prompts.totor import gameof24 as totor_prompts
from src.prompts.adapt import gameof24 as llama_prompts
from src.states.gameof24 import GameOf24State
from utils import parse_suggestions, create_box


class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api, namespace)-> GameOf24State:
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

            # Set up CoT prompt
            if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
                prompt = llama_prompts.cot_prompt.format(input=state.puzzle) + "Steps:\n" + steps + "Answer: "
            else:
                prompt = totor_prompts.cot_prompt.format(input=state.puzzle) + "Steps:\n" + steps

            # Get the final expression
            suggestions = await api.buffered_request(prompt, key=hash(state), namespace=namespace)

            # State does not change, only the steps
            selected_suggestion = "Answer: " + suggestions
            selected_state = state.current_state
        else:
            # Set up BFS prompt
            if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
                prompt = llama_prompts.bfs_prompt.format(input=current_state)
            else:
                prompt = totor_prompts.bfs_prompt.format(input=current_state)

            # Get the next state
            suggestions = await api.buffered_request(prompt, key=hash(state), namespace=namespace)

            # parse suggestions, based on the current state
            parsed_suggestions = parse_suggestions(suggestions)
            if parsed_suggestions == []:
                print(f"State: {state}")
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
    async def evaluate(state: GameOf24State, api, value_cache, namespace, n=3):
        last_step = state.steps[-1]
        
        # Should not happen
        if "left" not in last_step:
            answer = last_step.lower().replace("answer: ", "")

            if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
                prompt = llama_prompts.value_last_step_prompt.format(input=state.puzzle, answer=answer)
            else:
                prompt = totor_prompts.value_last_step_prompt.format(input=state.puzzle, answer=answer)

            error = f"""Current state '{state.current_state}'\nSteps: '{" -> ".join(state.steps)}'"""
            print(f"Evaluating terminal state that is not correct : {state}")
            return 0
        else:
            if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
                prompt = llama_prompts.value_prompt.format(input=state.current_state)
            else:
                prompt = totor_prompts.value_prompt.format(input=state.current_state)

        if prompt in value_cache:
            value_number = value_cache[prompt]
        
        else:
            coroutines = []
            for _ in range(n):
                coroutines.append(api.buffered_request(prompt, key=hash(state), namespace=namespace))
            iid_replies = await asyncio.gather(*coroutines)

            # Unwrap the iid_replies

            if len(state.steps) == 4 and 'answer' not in "\n".join(state.steps).lower():
                value_number = 0
            
            else:
                value_names = [value.split('\n')[-1] for value in iid_replies]
                value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
                value_number = sum(value * value_names.count(name) for name, value in value_map.items())
            value_cache[prompt] = value_number
        
        return value_number

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
