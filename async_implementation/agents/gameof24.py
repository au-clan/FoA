import asyncio
import re
import math
import random
import numpy as np
from sympy import simplify

random.seed(0)

from async_implementation.prompts import gameof24 as prompts
from async_implementation.states.gameof24 import GameOf24State
from async_implementation.resampling import value_weighted


class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api)-> GameOf24State:
        """
        Given a state, returns the next state (1-to-1).
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
            prompt = prompts.cot_prompt.format(input=state.puzzle) + "\n" + steps

            # Get the final expression
            suggestions = await api.buffered_request(prompt, key=hash(state))

            # State does not change, only the steps
            selected_suggestion = suggestions
            selected_state = state.current_state
        else:
            # BFS prompt
            prompt = prompts.bfs_prompt.format(input=current_state)

            # Get the next state
            suggestions = await api.buffered_request(prompt, key=hash(state))

            # parse suggestions, based on the current state
            suggestions = suggestions.split("\n")
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
    async def evaluate(state: GameOf24State, api, n=3):
        last_step = state.steps[-1]
        if "left" not in last_step:
            answer = last_step.lower().replace("answer: ", "")
            prompt = prompts.value_last_step_prompt.format(input=state.puzzle, answer=answer)
        else:
            prompt = prompts.value_prompt.format(input=state.current_state)

        coroutines = []
        for _ in range(n):
            coroutines.append(api.buffered_request(prompt, key=hash(state)))
        iid_replies = await asyncio.gather(*coroutines)
        value_names = [value.split('\n')[-1] for value in iid_replies]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value_number = sum(value * value_names.count(name) for name, value in value_map.items())
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
            """
            expression = state.steps[-1].lower().replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', state.puzzle)
            if sorted(numbers) != sorted(problem_numbers):
                return {'r': 0}
            try:
                return {'r': int(simplify(expression) == 24)}
            except Exception as e:
                # print(e)
                return {'r': 0}
    
    @staticmethod
    def resample(values: list, n_picks: int, randomness: int, resampling_method: str="linear", percentile: float=0.75)-> list:

        methods = {
            "linear": value_weighted.linear,
            "logistic": value_weighted.logistic,
            "max": value_weighted.max,
            "percentile": value_weighted.percentile
        }

        if resampling_method not in methods:
            raise ValueError(f"Invalid resampling method: {resampling_method}\nValid methods: {methods.keys()}")
        
        probabilities = methods[resampling_method](values)
        random.seed(randomness)
        randomness = random.randint(0, 1000)
        resampled_indices = np.random.choice(range(len(values)), size=n_picks, p=probabilities, replace=True)
        return resampled_indices.tolist()

