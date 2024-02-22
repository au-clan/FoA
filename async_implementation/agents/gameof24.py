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
    async def step(state: GameOf24State, api, limiter, n: int=1):
        """
        Given a state, returns n next states.
        """

        # set up the prompt, based on the current state

        # ToT uses bfs_prompt to generate next steps but then uses
        # the cot_prompt to get the final expression. 
        # For example, input : 1 1 4 6
        # Step 0 : '1 - 1 = 0 (left: 0 4 6)'          BFS prompt
        # Step 1 : '0 + 4 = 4 (left: 4 6)'            BFS prompt
        # Step 2 : '4 * 6 = 24 (left: 24)'            BFS prompt
        # Step 3 : Answer : ((1 - 1) + 4) * 6 = 24    CoT prompt


        suggestions  = []

        # Answer not found -> Need to compute next step
        if state.current_state.strip() != "24":
            prompt = prompts.bfs_prompt.format(input=state.current_state)

            # Get the next state
            while len(suggestions) < n:
                messages = [{"role": "user", "content": prompt}]
                iid_suggestions = await api.request(messages, limiter, n=math.ceil(n/8)) # Prompt suggests 8 new states
                [suggestions.extend(suggestion.split("\n")) for suggestion in iid_suggestions]

            # parse suggestions
            random.seed(state.randomness)
            selected_suggestions = random.sample(suggestions, k=n)
            selected_states = [GameOf24Agent.parse_next_state(x) for x in selected_suggestions]

        # Answer found -> Need to compute final expression
        else:
            current_steps = '\n'.join(state.steps) + "\n"
            prompt = prompts.cot_prompt.format(input=state.puzzle) + "\n" + current_steps

            #  Get the final expression
            messages = [{"role": "user", "content": prompt}]
            iid_suggestions = await api.request(messages, limiter, n=n) 
            suggestions.extend(iid_suggestions)

            # We already computed exactly the number of suggestions needed
            selected_suggestions = suggestions

            # We don't change states
            selected_states = [state.current_state]*n

        # set up new state objects
        next_states = []
        for next_suggestion, next_state in zip(selected_suggestions, selected_states):
            next_state = GameOf24State(
                puzzle=state.puzzle,
                current_state=next_state,
                steps=state.steps + [next_suggestion],
                randomness=random.randint(0, 1000)
            )
            next_states.append(next_state)
        return next_states

    @staticmethod
    async def evaluate(state: GameOf24State, api, limiter, n=3):
        last_step = state.steps[-1]
        if "left" not in last_step:
            answer = last_step.lower().replace("answer: ", "")
            prompt = prompts.value_last_step_prompt.format(input=state.puzzle, answer=answer)
        else:
            prompt = prompts.value_prompt.format(input=state.current_state)
        messages = [{"role": "user", "content": prompt}]
        iid_replies = await api.request(messages, limiter, n=3)
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

    class Resampling:

        @staticmethod
        def linear( values: list, n_picks: int, randomness: int)-> list:
            """
            Inputs:
                values: Initial values.
                n_picks: Number of picks to return.
                randomness: Current state of randomness.
            Outputs:
                resampled_indices: Indices of picked values.
            """
            probabilities = value_weighted.linear(values)
            random.seed(randomness)
            randomness = random.randint(0, 1000)
            resampled_indices = np.random.choice(range(len(values)), size=n_picks, p=probabilities, replace=True)
            return resampled_indices

