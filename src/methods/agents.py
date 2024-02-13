import json, os
import numpy as np
from src.tasks.base import Task
from src.methods.resampler import Resampler

class Agents():
    def __init__(self, task: Task, idx_input: int, n_agents: int, init: bool, **kwargs):
        # Task and model names (for logging)
        self.task_name = task.__name__
        self.model_name = kwargs.get('model', 'none').__class__.__name__

        # Create agents, get input, create values list and set Resampler
        self.input_idx = idx_input
        self.agents = [task(**kwargs) for agent in range(n_agents)] 
        for agent in self.agents:
            agent.get_input(idx_input)
        self.values = np.array([None] * n_agents)
        self.resampler = Resampler()
        
        # Set input, max steps and start step count
        self.input = self.agents[0].input
        self.max_steps = self.agents[0].max_steps
        self.step_count = 0
        
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
        cost_before = self.agents[0].model.get_cost(verbose=False)
        for agent in self.agents:
            agent.step()
        self.step_count += 1
        cost_after = self.agents[0].model.get_cost(verbose=False)
        print(f"Step cost : {cost_after-cost_before} ({len(self.agents)} agents)")

        # Log steps
        self.log[self.input_idx][f"step_{self.step_count}"] = {}
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[self.input_idx][f"step_{self.step_count}"]['steps'] = current_steps


    def evaluate(self, n: int = 3):
        """
        All agents evaluate their current state.
        """
        cost_before = self.agents[0].model.get_cost(verbose=False)
        for i, agent in enumerate(self.agents):
            value = agent.evaluate(n=n)
            self.values[i] = value
        cost_after = self.agents[0].model.get_cost(verbose=False)
        print(f"Evaluation cost : {cost_after-cost_before} ({len(self.agents)} agents)")
        # Log values
        self.log[self.input_idx][f"step_{self.step_count}"]['values'] = self.values.tolist()


    def resample(self, resample_method: str = 'normalization'):
        """
        Given a resampling method, resample the agents based on their values.
        """
        # Indices of resampled agents
        indices = self.resampler.resample(self.values, resample_method)
        pre_resampling_states = [agent.get_state() for agent in self.agents]
        log_indices = []

        # The agents copy the state of the resampled agents
        for i, agent in enumerate(self.agents):
            agent.copy_state(pre_resampling_states[indices[i]])
            log_indices.append(f"{i} <- {indices[i]}")
        
        # Log resampling
        self.log[self.input_idx][f"step_{self.step_count}"]['resampled'] = log_indices
        current_steps = ["\n".join(agent.steps) for agent in self.agents]
        self.log[self.input_idx][f"step_{self.step_count}"]['resampled_steps'] = current_steps
    
    def test_output(self)-> list:
        results = [agent.test_output() for agent in self.agents]

        # Log results
        self.log[self.input_idx]["results"] = results
        self.log[self.input_idx]["cost"] = self.agents[0].model.get_cost(verbose=False)
        return results


    def choose(self)-> Task:
        """
        Returns the best agent best on their state's last evaluation.
        """
        best_agent_idx = np.argmax(self.values)
        best_agent = self.agents[best_agent_idx]
        best_answer = "n".join(best_agent.steps)
        print(f"Best answer: {best_answer}")

        # Log best agent
        self.log[self.input_idx]["best_agent"] = {}
        self.log[self.input_idx]["best_agent"]['idx'] = int(best_agent_idx)
        self.log[self.input_idx]["best_agent"]['steps'] = "\n".join(best_agent.steps)
        return best_agent

    def create_log(self, repo_path: str, file_name:str = None):
        """
        Given the repo path, create a log file (in the given repo).
        """
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

    




