import os, re, json, random
import pandas as pd
from copy import deepcopy
from sympy import simplify

from initial_implementation.tasks.base import Task, DATA_PATH
from initial_implementation.prompts.game24 import foa_step_prompt, cot_prompt, value_prompt, value_last_step_prompt, bfs_prompt

class Game24(Task):
    def __init__(self, model, file='24_tot.csv', foa_prompt:bool = False):
        super().__init__()
        path = os.path.join(DATA_PATH, file)
        self.data = pd.read_csv(path).Puzzles.tolist()
        self.current_state = None
        self.model = model
        self.steps = []
        self.input = None
        self.input_idx = None
        self.max_steps = 4
        self.steps_count = 0
        self.foa_prompt = foa_prompt

    def __len__(self) -> int:
        """
        Returns the number of examples (possible inputs) for the task.
        """
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        """
        Sets the input for the task given its idx.
        """
        if idx <0 or idx > self.__len__():
            raise IndexError(f'Index {idx} out of range for Game24 task with {len(self)} examples.')
        else:
            input = self.data[idx]
            self.input = input
            self.input_idx = idx
            self.current_state = input
            self.steps = []

    def step(self):
        """
        Performs a step in the game of 24 task
        """
        temp = self.current_state
        if self.current_state.strip() == "24":
            steps = '\n'.join(self.steps) + "\n"
            prompt = cot_prompt.format(input=self.input) + "\n" + steps
            suggestion = self.model.request(prompt)[0]
            self.steps.append(suggestion)
            print(f"Prompt: CoT, State : {temp}\n\tSuggestion: {suggestion}")
        
        elif self.foa_prompt:
            prompt = foa_step_prompt.format(input=self.current_state)
            suggestion = self.model.request(prompt)[0]
            self.current_state = self.get_current_state(suggestion)
            self.steps.append(suggestion)
            print(f"Prompt: FoA, State : {temp}\n\tSuggestion: {suggestion}")
        
        else:
            prompt = bfs_prompt.format(input=self.current_state)
            suggestions = self.model.request(prompt)[0]                         # bfs prompt produces multiple suggestions
            selected_suggestion = random.choice(suggestions.split("\n"))        # We select random suggestion
            self.current_state = self.get_current_state(selected_suggestion)
            self.steps.append(selected_suggestion)
            print(f"Prompt: BFS, State : {temp}\n\tSuggestion: {selected_suggestion}")

        
        self.steps_count += 1

    @staticmethod
    def get_current_state(suggestion: str) -> str:
        return suggestion.split('left: ')[-1].split(')')[0]

    def evaluate(self, n: int = 3)-> float:
        last_line = self.steps[-1]
        if 'left: ' not in last_line:  # last step
            answer = last_line.lower().replace('answer: ', '')
            prompt = value_last_step_prompt.format(input=self.input, answer=answer)
        else:
            prompt = value_prompt.format(input=self.current_state)
        response = self.model.request(prompt, n=n)
        value_names = [value.split('\n')[-1] for value in response]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value_number = sum(value * value_names.count(name) for name, value in value_map.items())
        return value_number
    
    def get_state(self)-> dict:
        """
        Collects the values that contribute towards the state of the task in a dictionary.
        """
        state = {"steps": self.steps, "current_state": self.current_state}
        return state
    
    def copy_state(self, state: dict):
        """
        Given a state (dictionary), copy the state values to the current task.
        """
        for key, value in state.items():
            setattr(self, key, deepcopy(value))

    # Taken from ToT
    def test_output(self)-> dict:
            """
            Tests the output of a given task
                1. Checks if the numbers used are the same as the ones provided.
                2. Checks if the operations performed result to 24.
            """
            expression = self.steps[-1].lower().replace('answer: ', '').split('=')[0]
            numbers = re.findall(r'\d+', expression)
            problem_numbers = re.findall(r'\d+', self.data[self.input_idx])
            if sorted(numbers) != sorted(problem_numbers):
                return {'r': 0}
            try:
                return {'r': int(simplify(expression) == 24)}
            except Exception as e:
                # print(e)
                return {'r': 0}
    
    @staticmethod
    def get_accuracy(log_path: str, verbose: bool=True)-> float:
        with open(log_path, "r") as log_file:
            log = json.load(log_file)
        
        correct_experiments = 0
        for experiment in log:
            result = log[experiment]["results"]
            if {"r":1}in result:
                correct_experiments+=1
        
        accuracy = correct_experiments/len(log)
        if verbose:
            print(f"Predicted correctly {correct_experiments}/{len(log)} ({accuracy*100:.2f}%)")
        return accuracy
    
    @staticmethod
    def get_cost(log_path: str, verbose: bool=True)-> float:
        with open(log_path, "r") as log_file:
            log = json.load(log_file)
            experiments_idx = [int(idx) for idx in log.keys()]
            last_experiment_idx = max(experiments_idx)
            cost = log[str(last_experiment_idx)]["cost"]
        
        if verbose:
            print(f"Experiment cost : {cost} USD")
        return cost


    @staticmethod
    def init_step(input:str, n:int, model)-> list:
        print("Init stepping")
        prompt = bfs_prompt.format(input=input)
        steps = []
        while len(steps)<n:
            response = model.request(prompt)[0].split("\n")
            unique_steps = [step for step in response if step not in steps] # Not using sets to keep order
            steps.extend(unique_steps)
        return steps[:n]
        

    
        