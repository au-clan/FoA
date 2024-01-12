import os
import pandas as pd
from src.tasks.base import Task, DATA_PATH
from src.prompts.game24 import foa_step_prompt, cot_prompt

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

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
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
            suggestion = self.model.request(prompt)
            self.steps.append(suggestion)
        
        else:
            prompt = foa_step_prompt.format(input=self.current_numbers)
            suggestion = self.model.request(prompt)
            self.current_numbers = self.get_current_numbers(suggestion)
            self.steps.append(suggestion)

    @staticmethod
    def get_current_numbers(suggestion: str) -> str:
        return suggestion.split('left: ')[-1].split(')')[0]
