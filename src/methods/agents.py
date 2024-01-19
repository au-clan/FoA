import json, os
import numpy as np
from src.tasks.base import Task
from src.methods.resampler import Resampler

class Agents():
    def __init__(self, task: Task, idx_input: int, n_agents: int, **kwargs):
        self.task_name = task.__name__
        self.input_idx = idx_input
        self.agents = [task(**kwargs) for agent in range(n_agents)] 
        for agent in self.agents:
            agent.get_input(idx_input)
        self.max_steps = self.agents[0].max_steps
        self.input = self.agents[0].input
        self.values = np.array([None] * n_agents)
        self.resampler = Resampler()
        self.step_count = 0
        self.log = {}
        self.log['input'] = self.input


    def __getitem__(self, idx: int)-> Task:
        return self.agents[idx]
    

    def __len__(self):
        return len(self.agents)
    

    def step(self):
        for agent in self.agents:
            agent.step()
        self.step_count += 1

        # Log steps
        self.log[f"step_{self.step_count}"] = {}
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[f"step_{self.step_count}"]['steps'] = current_steps


    def evaluate(self, n: int = 3):
        for i, agent in enumerate(self.agents):
            value = agent.evaluate(n=n)
            self.values[i] = value
        
        # Log values
        self.log[f"step_{self.step_count}"]['values'] = self.values.tolist()


    def resample(self, resample_method: str = 'normalization'):
        indices = self.resampler.resample(self.values, resample_method)
        log_indices = []
        for i, agent in enumerate(self.agents):
            agent.copy(self.agents[indices[i]])
            log_indices.append(f"{i} <- {indices[i]}")
        
        # Log resampling
        self.log[f"step_{self.step_count}"]['resampled'] = log_indices
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[f"step_{self.step_count}"]['resampled_steps'] = current_steps

    def create_log(self, repo_path: str):
        file_name = f"{self.task_name}_{self.input_idx}.json"
        file_path = os.path.join(repo_path, file_name)
        with open(file_path, 'w+') as f:
            json.dump(self.log, f, indent=4)




