from src.tasks.base import Task

class Agents():
    def __init__(self, task: Task, idx: int, n_agents: int, **kwargs):
        self.agents = {i: task(**kwargs) for i in range(n_agents)}
        for agent in self.agents.values():
            agent.get_input(idx)
        self.max_steps = self.agents[0].max_steps
        self.input = self.agents[0].input

    def __getitem__(self, idx):
        return self.agents[idx]
    
    def __len__(self):
        return len(self.agents)
    
    def step(self):
        for agent in self.agents.values():
            agent.step()

    def evaluate(self, n: int = 3):
        for agent in self.agents.values():
            agent.evaluate(n=n)