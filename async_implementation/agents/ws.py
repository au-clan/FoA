import gym
import random
from uuid import uuid4

from bs4 import BeautifulSoup
from bs4.element import Comment

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
        self.env = gym.make('WebAgentTextEnv-v0', observation_mode='html', server=server)
        obs, infos = self.env.reset(session=env_id)
        self.action_history.append("reset")
        self.rewards.append(0)
        self.observations.append(self.parse_obs(obs))
        self.infos.append(infos)

        self.terminal = False

        if prompt == "act":
            self.init_prompt = prompts.actonly_prompt
        elif prompt == "react":
            self.init_prompt = prompts.react_prompt
        else:
            raise ValueError(f"Unknown prompt type: {prompt} (should be 'act' or 'react')")
        self.prompt = f"{self.parse_obs(obs)}\n\nAction:"

        # If the first action was to reset skip.
        for action in replay_actions:
            obs, reward, terminal, infos = self.env.step(action)
            self.terminal = terminal
            self.observations.append(obs)
            self.rewards.append(reward)
            self.infos.append(infos)
            self.action_history.append(action)

    async def step(self, api, namespace):
        assert len(self.observations) == len(self.infos)
        assert len(self.observations) == len(self.action_history)
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
            self.rewards.append(self.rewards[-1])
            self.infos.append(self.infos[-1])

        if action.startswith("think"):
            obs = "OK."
        
        
        self.action_history.append(action)

        if obs not in ["OK.", "Invalid action!"]:
            obs = self.parse_obs(obs)
        self.observations.append(obs)
        self.prompt += f"{action}\nObservation: {obs}\n\nAction:"





    
    async def clone(self, random_seed):
        cloned_agent = WebShopAgent(self.env_id, random_seed, self.server, replay_actions=self.action_history[1:])
        assert cloned_agent.observations == self.observations, "cloned agent should have the same observations as the original agent"
        assert cloned_agent.infos == self.infos, "cloned agent should have the same infos as the original agent"
        assert cloned_agent.action_history == self.action_history, "cloned agent should have the same action history as the original agent"
        assert not cloned_agent.terminal, "it doesn't make sense to clone a terminal agent, this points to a logic error in the outer algorithm"
        return cloned_agent
    
    def hash(self):
        return hash((self.env_id, " ".join(self.observations), " -> ".join(self.action_history), self.random_seed))

    def parse_obs(self, obs):
        print(f"ACTION HISTORY : {self.action_history}")
        if self.action_history[-1] in ["reset", "click[Back to Search]"]:
            page_type = "init"
        else:
            page_type = "not_init"
        
        print(f"PAGE: {page_type}")
        html = obs
        html_obj = BeautifulSoup(html, 'html.parser')
        texts = html_obj.findAll(text=True)
        visible_texts = list(filter(tag_visible, texts))
        if False:
            # For `simple` mode, return just [SEP] separators
            return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
        else:
            # Otherwise, return an observation with tags mapped to specific, unique separators
            observation = ''
            option_type = ''
            options = {}
            asins = []
            cnt = 0
            prod_cnt = 0
            just_prod = 0
            for t in visible_texts:
                if t == '\n': continue
                if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
                if t.parent.name == 'button':  # button
                    processed_t = f'\n[{t}] '
                elif t.parent.name == 'label':  # options
                    if False: #f"'{t}'" in url: (No such cases in ReAcz)
                        processed_t = f'[[{t}]]'
                        # observation = f'You have clicked {t}.\n' + observation
                    else:
                        processed_t = f'[{t}]'
                    options[str(t)] = option_type
                    # options[option_type] = options.get(option_type, []) + [str(t)]
                elif t.parent.get('class') == ["product-link"]: # product asins
                    processed_t = f'\n[{t}] '
                    if prod_cnt >= 3:
                        processed_t = ''
                    prod_cnt += 1
                    asins.append(str(t))
                    just_prod = 0
                else: # regular, unclickable text
                    processed_t =  '\n' + str(t) + ' '
                    if cnt < 2 and page_type != 'init': processed_t = ''
                    if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                    option_type = str(t)
                    cnt += 1
                just_prod += 1
                observation += processed_t
            info = {}
            if options:
                info['option_types'] = options
            if asins:
                info['asins'] = asins
            if 'Your score (min 0.0, max 1.0)' in visible_texts:
                idx = visible_texts.index('Your score (min 0.0, max 1.0)')
                info['reward'] = float(visible_texts[idx + 1])
                observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
            return clean_str(observation)

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )