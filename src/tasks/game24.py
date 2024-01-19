import os, copy
import pandas as pd
from src.tasks.base import Task, DATA_PATH
from src.prompts.game24 import foa_step_prompt, cot_prompt, value_prompt, value_last_step_prompt

class Game24(Task):
    def __init__(self, model, file='24_tot.csv'):
        super().__init__()
        path = os.path.join(DATA_PATH, file)
        self.data = pd.read_csv(path).Puzzles.tolist()
        self.current_numbers = None
        self.model = model
        self.steps = []
        self.input = None
        self.max_steps = 4
        self.steps_count = 0
        self.values_log = {}

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
            self.current_numbers = input
            self.input = input
            self.steps = []

    def step(self):
        if self.current_numbers.strip() == "24":
            steps = '\n'.join(self.steps) + "\n"
            prompt = cot_prompt.format(input=self.input) + "\n" + steps
            suggestion = self.model.request(prompt)[0]
            self.steps.append(suggestion)
        
        else:
            prompt = foa_step_prompt.format(input=self.current_numbers)
            suggestion = self.model.request(prompt)[0]
            self.current_numbers = self.get_current_numbers(suggestion)
            self.steps.append(suggestion)
        
        self.steps_count += 1

    @staticmethod
    def get_current_numbers(suggestion: str) -> str:
        return suggestion.split('left: ')[-1].split(')')[0]

    def evaluate(self, n: int = 3)-> float:
        last_line = self.steps[-1]
        if 'left: ' not in last_line:  # last step
            answer = last_line.lower().replace('answer: ', '')
            prompt = value_last_step_prompt.format(input=self.input, answer=answer)
        else:
            prompt = value_prompt.format(input=self.current_numbers)
        response = self.model.request(prompt, n=n)
        value_names = [value.split('\n')[-1] for value in response]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value_number = sum(value * value_names.count(name) for name, value in value_map.items())
        self.values_log[self.steps_count] = value_number
        return value_number
    
    def copy(self, task: Task):
        self.steps = task.steps.copy()
        self.current_numbers = task.current_numbers
        self.values_log = task.values_log.copy()
        