import re
import random
from sympy import simplify

random.seed(0)

from async_implementation.prompts import gameof24 as prompts
from async_implementation.states.gameof24 import GameOf24State


class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api, limiter):
        """
        Given a state, return the next state.
        """

        # set up the prompt, based on the current state

        # ToT uses bfs_prompt to generate next steps but then uses
        # the cot_prompt to get the final expression. 
        # For example, input : 1 1 4 6
        # Step 0 : '1 - 1 = 0 (left: 0 4 6)'          BFS prompt
        # Step 1 : '0 + 4 = 4 (left: 4 6)'            BFS prompt
        # Step 2 : '4 * 6 = 24 (left: 24)'            BFS prompt
        # Step 3 : Answer : ((1 - 1) + 4) * 6 = 24    CoT prompt

        current_state = state.current_state

        # Answer not found -> Need to compute next step
        if current_state.strip() != "24":
            prompt = prompts.bfs_prompt.format(input=current_state)

            # Get the next state
            messages = [{"role": "user", "content": prompt}]
            iid_suggestions = await api.request(messages, limiter, n=1)
            suggestions = iid_suggestions[0]

            # parse suggestions
            suggestions = suggestions.split("\n")
            random.seed(state.randomness)
            selected_suggestion = random.choice(suggestions)
            selected_state = GameOf24Agent.parse_next_state(selected_suggestion)

        # Answer found -> Need to compute final expression
        else:
            current_steps = '\n'.join(state.steps)
            prompt = prompts.cot_prompt.format(input=current_state) + "\n" + current_steps

            #  Get the final expression
            messages = [{"role": "user", "content": prompt}]
            iid_suggestions = await api.request(messages, limiter, n=1)
            suggestions = iid_suggestions[0]

            # Only 1 suggestion (because of prompt)
            selected_suggestion = suggestions

            # We don't change states
            selected_state = current_state

        

        # set up new state object
        next_state = GameOf24State(
            puzzle=state.puzzle,
            current_state=selected_state,
            steps=state.steps + [selected_suggestion],
            randomness=random.randint(0, 1000)
        )
        return next_state

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

