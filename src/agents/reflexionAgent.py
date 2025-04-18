from src.states.gameof24 import GameOf24State
from src.prompts.adapt import gameof24 as llama_prompts
import random
from utils import parse_suggestions, create_box
import re
from sympy import simplify
from typing import Any, Dict, List, Tuple

class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api, namespace, reflexion: List[str])-> GameOf24State:
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
                prompt = llama_prompts.bfs_prompt_single.format(input=current_state) 
            else:
                prompt = llama_prompts.bfs_reflexion_prompt_single.format(input=current_state, reflexion=reflexion[0]) 

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
    def generate_reflexion(time_of_reflexion: str, puzzle: str, steps: List[str], state: GameOf24State, api, namespace) -> str:
        """
        Generates a reflexion based on the puzzle and the steps done
        """
        if time_of_reflexion == "step_wise":
            prompt = llama_prompts.reflexion_prompt.format(puzzle=puzzle, steps=steps)
        else:
            prompt = llama_prompts.new_reflexion_prompt.format(puzzle=puzzle, steps=steps)
        reflexion = api.buffered_request(prompt, key=hash(state), namespace=namespace)
        return reflexion
    

    @staticmethod
    async def evaluate_step(puzzle: str, steps: List[str], state: GameOf24State, api, namespace: tuple)-> str:
        """
        Uses the evaluate prompt which asks the model to determine if a step is invalid or infeasible and return the step.
        """
        prompt = llama_prompts.evaluate_prompt.format(puzzle=puzzle, steps=steps)
        evalution = await api.buffered_request(prompt, key=hash(state), namespace=namespace)
        return evalution


    def validate(puzzle: str, steps: List[str], state: GameOf24State, api, namespace) -> str:
        """
        Uses the validation prompt to check if the steps are valid.
        """
        validation_prompt = llama_prompts.validation_prompt.format(puzzle=puzzle, steps=steps)
        validation = api.buffered_request(validation_prompt, key=hash(state), namespace=namespace)
        return validation
    

    def value(puzzle: str, steps: List[str], state: GameOf24State, api, namespace) -> str:
        """
        Uses the value prompt to estimate the feasibility of the steps.
        """
        input = "{state} \n {steps[0]}\n {steps[1]}\n {steps[2}\n {steps[3]}\n"
        value_prompt = llama_prompts.value_prompt.format(input=input)
        value = api.buffered_request(value_prompt, key=hash(state), namespace=namespace)
        return value


    @staticmethod
    def generate_summary(reflexion: List[str], state: GameOf24State, api, namespace) -> str:
        """
        Generates a summary of current reflexions
        """
        prompt = llama_prompts.summary_prompt.format(reflexion=reflexion)
        reflexion = api.buffered_request(prompt, key=hash(state), namespace=namespace)
        return reflexion
