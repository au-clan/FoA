import json, os
import numpy as np
from initial_implementation.tasks.base import Task
from initial_implementation.methods.resampler import Resampler

class Agents():
    def __init__(self, task: Task, idx_input: int, n_agents: int, init: bool, back_coef: float=0.8, n_evaluations: int=3, **kwargs):
        # Task and model names (for logging)
        self.task_name = task.__name__
        self.model_name = kwargs.get('model', 'none').__class__.__name__
        

        # Create agents and get input 
        self.input_idx = idx_input
        self.agents = [task(**kwargs) for agent in range(n_agents)] 
        for agent in self.agents:
            agent.get_input(idx_input)

        # Set input, max steps and start step count
        self.input = self.agents[0].input
        self.max_steps = self.agents[0].max_steps
        self.step_count = 0

        # Create values list and set Resampler
        self.values = {"idx":["INIT"], "values":[n_evaluations*20], "states":[{"steps":[], "current_state":self.input}]}
        self.n_evaluations =n_evaluations
        self.resampler = Resampler()
        self.back_coef = back_coef
        self.n_evaluations = n_evaluations
        
        # Start logging
        self.log = {self.input_idx:{}}
        self.log[self.input_idx]['input'] = self.input

        # Initialization
        if init:
            print("Random initialization")
            self.step_count = 1
            steps = task.init_step(input=self.input, n=len(self.agents), model=self.agents[0].model)
            for step, agent in zip(steps, self.agents):
                agent.step_count = 1
                agent.steps.append(step)
                agent.current_state = task.get_current_state(step)
            
            # Log steps
            self.log[self.input_idx][f"step_{self.step_count}"] = {}
            current_state = ["\n".join(agent.steps) for agent in self.agents]
            self.log[self.input_idx][f"step_{self.step_count}"]['steps'] = current_state


    def __getitem__(self, idx: int)-> Task:
        """
        Given an index, return the agent at that index.
        """
        return self.agents[idx]
    

    def __len__(self):
        """
        Return the number of agents.
        """
        return len(self.agents)
    

    def step(self):
        """
        All agents take a step.
        """
        for agent in self.agents:
            agent.step()
        self.step_count += 1

        # Decade value of previous states every time a step is performed
        old_state_values = np.array(self.values["values"])
        old_state_new_values = (self.back_coef * old_state_values).tolist()
        old_state_new_values[0] = old_state_new_values[0] * self.back_coef
        self.values["values"] = old_state_new_values
        

        # Log steps
        self.log[self.input_idx][f"step_{self.step_count}"] = {}
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[self.input_idx][f"step_{self.step_count}"]['steps'] = current_steps


    def evaluate(self, n: int = 3):
        """
        All agents evaluate their current state.
        """
        for i, agent in enumerate(self.agents):
            self.values["idx"].append(f"{self.step_count}.{i}")
            self.values["values"].append(agent.evaluate(n=n))
            self.values["states"].append(agent.get_state())

        # Log values
        self.log[self.input_idx][f"step_{self.step_count}"]['values'] = self.values["values"][-len(self.agents):]


    def resample(self, resample_method: str = 'normalization'):
        """
        Given a resampling method, resample the agents based on their values.
        """
        # Indices of resampled agents
        indices = self.resampler.resample(np.array(self.values["values"]), resample_method)
        selected_states = [self.values["states"][i] for i in indices]
        selected_states_idx = [self.values["idx"][i] for i in indices]
        log_indices = []

        # The agents copy the state of the resampled agents
        for i, agent, in enumerate(self.agents):
            agent.copy_state(selected_states[i])
            print(f"{i} <- {selected_states_idx[i]}")
            log_indices.append(f"{i} <- {selected_states_idx[i]}")
        
        # Log resampling
        self.log[self.input_idx][f"step_{self.step_count}"]['resampled'] = log_indices
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[self.input_idx][f"step_{self.step_count}"]['resampled_steps'] = current_steps
    
    def test_output(self)-> tuple:
        results = [agent.test_output() for agent in self.agents]
        done = {"r":1} in results

        # Log results
        self.log[self.input_idx]["results"] = results
        self.log[self.input_idx]["cost"] = self.agents[0].model.get_cost(verbose=False)
        return done, results

    def create_log(self, repo_path: str, file_name:str = None):
        """
        Given the repo path, create a log file (in the given repo).
        """
        # Moving results and cost log at the end
        results = self.log[self.input_idx].pop("results")
        self.log[self.input_idx]["results"] = results
        cost = self.log[self.input_idx].pop("cost")
        self.log[self.input_idx]["cost"] = cost

        if not file_name:
            file_name = f"{self.task_name}_{self.input_idx}inputIdx_{len(self.agents)}agents_{self.model_name}.json"
        file_path = os.path.join(repo_path, file_name)

        if os.path.exists(file_path):
            with open(file_path, "r") as log_file:
                log = json.load(log_file)
        else:
            log = {}
        
        log.update(self.log)

        with open(file_path, 'w+') as f:
            json.dump(log, f, indent=4)

    




