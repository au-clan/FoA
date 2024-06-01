from uuid import uuid4

from bs4.element import Comment

import async_implementation.prompts.ws as prompts
from utils import webshopEnv

class WebShopAgent:
    def __init__(self, env_id, random_seed, id=None, replay_actions=[], values=[], prompting="react"):
        self.unique_id = uuid4()
        self.random_seed = random_seed
        self.observations = []
        self.rewards = []
        self.values = values
        self.prompt=None
        self.id = id

        self.env_id = env_id
        self.env = webshopEnv()

        if prompting == "act":
            self.init_prompt = prompts.actonly_prompt
        elif prompting == "react":
            self.init_prompt = prompts.react_prompt
        else:
            raise ValueError(f"Unknown prompt type: {prompting} (should be 'act' or 'react')")
        self.prompting = prompting

        # If the first action was to reset skip.
        if replay_actions:
            self.action_history = replay_actions
        else:
            self.action_history = ["reset"]
        self.step()
    
    def reset(self):
        obs, reward, done = self.env.step(session=f"fixed_{self.env_id}", action="reset")

        assert obs is not None, "Observation is None after reset"
        assert reward == 0, "Reward is not 0 after reset"
        assert not done, "Done is True after reset"

        self.observations = [obs]
        self.rewards = [0]
        self.terminal = False

        return obs, reward, done

    async def get_next_action(self, api, namespace):
        assert len(self.observations) == len(self.action_history)
        assert len(self.observations) > 0, "resetting the environment must generate one observation at the beginning"

        # Get the prompt
        prompt = self.get_complete_prompt(type="step")
        response = await api.buffered_request(prompt, key=self.hash(), namespace=namespace)
        action = response.strip(' ')
        self.action_history.append(action)
    
    def step(self):
        """
        This function assumes that the first action is a reset action. This is needed in our implementation.
        """
        self.observations = []
        self.rewards = []
        prompt=""
        for i, action in enumerate(self.action_history):
            
            obs, reward, terminal = self.take_action(action)
            self.update(obs, reward, terminal)
            
            if i == 0:
                prompt += f"{obs}\n\nAction:"
            else:
                prompt += f' {action}\nObservation: {obs}\n\nAction:'

        # Update prompt
        self.prompt = prompt


    def update(self, obs, reward, terminal):
        self.observations.append(obs)
        self.rewards.append(reward)
        self.terminal=terminal


    def take_action(self, action):
            
        # Try to impelement given action
        try:
            obs, reward, terminal = self.env.step(session=f"fixed_{self.env_id}", action=action)
        except AssertionError:
            obs = "Invalid action!"
            reward = 0
            terminal = False
        
        # Debugging
        except KeyError:
            print(f"KeyError in action: {action}")
            print(f"Prompt: {self.prompt}")
            print(f"Observations: {self.observations}")
            print(f"Action history: {self.action_history}")
            exit()
        assert obs is not None, f"Observation is None after action {action}"

        # If the action is a think action, adjust the observation
        if action.startswith("think"):
            obs = "OK."     
        
        return obs, reward, terminal
    
    
    async def evaluate(self, api, value_cache, namespace, verbose=False):
        prompt = self.get_complete_prompt(type="eval")
        
        if prompt in value_cache:
            value = value_cache[prompt]
        elif self.observations[-1] == "Invalid action!":
            value = 0
            value_cache[prompt] = value
        else:
            response = await api.buffered_request(prompt, key=self.hash(), namespace=namespace)
            if verbose:
                print(response)
            value = value_outputs_unwrap(response)
            value_cache[prompt] = value
        
        self.values.append(value)
        
    def clone(self, random_seed=None, id=None, allow_terminal=False):
        # If a random seed is not provided, clone an exact replica of the agent
        if random_seed is None:
            random_seed = self.random_seed
        
        # If an id is not provided, clone the agent with the same id
        if id is None:
            id = self.id
        
        # Clone the agent
        cloned_agent = WebShopAgent(self.env_id, random_seed, id=id, replay_actions=self.action_history.copy(), values=self.values.copy(), prompting=self.prompting)

        assert cloned_agent.observations == self.observations, "cloned agent should have the same observations as the original agent"
        assert cloned_agent.action_history == self.action_history, "cloned agent should have the same action history as the original agent"
        if not allow_terminal:
            assert not cloned_agent.terminal, "it doesn't make sense to clone a terminal agent, this points to a logic error in the outer algorithm"
        return cloned_agent
    
    def hash(self):
        return hash((self.env_id, " ".join(self.observations), " -> ".join(self.action_history)))

    
    def get_complete_prompt(self, type=None):
        step_prompt = self.init_prompt + self.prompt[-(6400-len(self.init_prompt)):]
        eval_prompt = prompts.score_prompt.format(s="", input=step_prompt + "\n\nReflection: ")

        if type is None:
            return {"step": step_prompt, "evaluate": eval_prompt}
        elif type == "step":
            return step_prompt
        elif type == "eval":
            return eval_prompt
        else:
            raise ValueError(f"Unknown prompt type: {type}")

# LATS: https://arxiv.org/abs/2310.04406
def value_outputs_unwrap(evaluate_prompt: str):
        if '10' in evaluate_prompt:
            return 1.0
        elif '9' in evaluate_prompt:
            return 0.9
        elif '8' in evaluate_prompt:
            return 0.8
        elif '7' in evaluate_prompt:
            return 0.7
        elif '6' in evaluate_prompt:
            return 0.6
        elif '5' in evaluate_prompt:
            return 0.5
        elif '4' in evaluate_prompt:
            return 0.4
        elif '3' in evaluate_prompt:
            return 0.3
        elif '2' in evaluate_prompt:
            return 0.2
        elif '1' in evaluate_prompt:
            return 0.1
        else:
            return 0.0