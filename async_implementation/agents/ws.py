import gym
import random
from uuid import uuid4

import async_implementation.prompts.ws_react as prompts

class WebShopAgent:
    def __init__(self, env_id, random_seed, server, replay_actions=[], prompt="react"):
        self.unique_id = uuid4()
        self.random_seed = random_seed
        self.observations = []
        self.rewards = []
        self.infos = []
        self.action_history = []

        self.env_id = env_id
        self.env = gym.make('WebAgentTextEnv-v0', observation_mode='text', server=server)
        obs, infos = self.env.reset(session=env_id)
        self.observations.append(obs)
        self.infos.append(infos)

        self.terminal = False

        if prompt == "act":
            self.init_prompt = prompts.actonly_prompt
        elif prompt == "react":
            self.init_prompt = prompts.react_prompt
        else:
            raise ValueError(f"Unknown prompt type: {prompt} (should be 'act' or 'react')")
        self.prompt = f"{obs}\n\nAction:"

        for action in replay_actions:
            obs, reward, terminal, infos = self.env.step(action)
            self.terminal = terminal
            self.observations.append(obs)
            self.rewards.append(reward)
            self.infos.append(infos)
            self.action_history.append(action)

    async def step(self, api, namespace):
        assert len(self.observations) == len(self.infos)
        assert len(self.observations) == len(self.action_history) + 1
        assert len(self.observations) > 0, "resetting the environment must generate one observation at the beginning"

        
        
        

        # Get next action from the system
        prompt = self.init_prompt + self.prompt[-(6400-len(self.init_prompt)):]
        response = await api.buffered_request(prompt, key=self.hash(), namespace=namespace)
        action = response.lstrip(' ')

        # Apply the action to the environment
        try:
            obs, reward, terminal, infos = self.env.step(action)
            self.terminal = terminal
            self.rewards.append(reward)
            self.infos.append(infos)
        except AssertionError:
            obs = "Invalid action!"

        if action.startswith("think"):
            obs = "OK."
            self.rewards.append(self.rewards[-1])
            self.infos.append(self.infos[-1])
        
        
        self.observations.append(obs)
        self.action_history.append(action)
        self.prompt += f"{action}\nObservation: {obs}\n\nAction:"





    
    async def clone(self, random_seed):
        cloned_agent = WebShopAgent(self.env_id, random_seed, self.server, replay_actions=self.action_history)
        assert cloned_agent.observations == self.observations, "cloned agent should have the same observations as the original agent"
        assert cloned_agent.infos == self.infos, "cloned agent should have the same infos as the original agent"
        assert cloned_agent.action_history == self.action_history, "cloned agent should have the same action history as the original agent"
        assert not cloned_agent.terminal, "it doesn't make sense to clone a terminal agent, this points to a logic error in the outer algorithm"
        return cloned_agent
    
    def hash(self):
        return hash((self.env_id, " ".join(self.observations), " -> ".join(self.action_history), self.random_seed))
