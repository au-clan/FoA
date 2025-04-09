import os
import smtplib
import json
import asyncio
import aiohttp

import numpy as np
from scipy.stats import bootstrap

from pathlib import Path
import pickle

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment




def create_folder(folder_path: str):
    """
    Creates a folder if it doesn't exist
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)


def delete_file(file_path: str):
    """
    Deletes a file given its path, if it already exists
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted : '{file_path}'")


def sigmoid(x):
    "Sigmoid function"
    return 1 / (1 + np.exp(-x))

def compare_json_files(file_path1, file_path2):
    """
    Just for debugging purposes. Compares two JSON files and returns True if they are the same, False otherwise.
    """
    try:
        with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
            data1 = json.load(file1)
            data2 = json.load(file2)
            
            return data1 == data2
    except FileNotFoundError:
        print("One or both files not found.")
        return False
    except json.JSONDecodeError:
        print("One or both files is not valid JSON.")
        return False
    
def create_box(input_string):
    """
    Creates a box around the input string and returns it.
    """
    lines = input_string.split('\n')
    max_length = max(len(line) for line in lines)
    border = '+' + '-' * (max_length + 2) + '+'

    boxed_string = border + '\n'
    for line in lines:
        boxed_string += '| ' + line.ljust(max_length) + ' |\n'
    boxed_string += border

    return boxed_string

def remove_after_last_bracket(input_string):
    """
    Given a string it removes everything after its last closing bracket ')
    """
    last_bracket_index = input_string.rfind(')')
    if last_bracket_index != -1:
        return input_string[:last_bracket_index + 1]  # Include the last bracket
    else:
        return input_string
    
def parse_suggestions(suggestions):

    # The prompt can potentially be cropped due to token constraints.
    # Therefore we remove everything after the last closing bracket ')'.

    suggestions = remove_after_last_bracket(suggestions)
    #valid_suggestions = [suggestion for suggestion in new_suggestions.split("\n") if "(left:" in suggestion]
    return suggestions.split("\n")
    
    
def update_actual_cost(api):
    """
    Used to track the actual cost of the API usage.
    """
    try:
        with open('actual_cost.txt', 'r') as file:
            current_cost = float(file.read())
    except FileNotFoundError:
        current_cost = 0  # If the file doesn't exist yet

    try:
        with open('groq_requests.txt', 'r') as file:
            current_requests = float(file.read())
    except FileNotFoundError:
        current_requests = 0  # If the file doesn't exist yet
    
    api_cost = api.cost(actual_cost=True)["total_cost"]
    new_cost = current_cost + api_cost

    api_requests = api.groq_requests
    new_requests = current_requests + api_requests

    with open('actual_cost.txt', 'w') as file:
        file.write(str(new_cost))
    
    with open('groq_requests.txt', 'w') as file:
        file.write(str(new_requests))

    return


WEBSHOP_URL = "http://10.90.38.15:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}


async def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def clean_str_sync(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

async def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (f'{WEBSHOP_URL}/{session}')
    if page_type == 'search':
        url = (f'{WEBSHOP_URL}/search_results/{session}/{query_string}/{page_num}')
    elif page_type == 'item':
        url = (f'{WEBSHOP_URL}/item_page/{session}/{asin}/{query_string}/{page_num}/{options}')
    elif page_type == 'item_sub':
        url = f'{WEBSHOP_URL}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{options}'
    elif page_type == 'end':
        url = f'{WEBSHOP_URL}/done/{session}/{asin}/{options}'

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            html = await response.text()
    
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
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
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

        return clean_str_sync(observation), info


def webshop_text_sync(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
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
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
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
        return clean_str_sync(observation), info


class webshopEnv:
  def __init__(self):
    self.sessions = {}
  
  async def step(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = await webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
  
  def step_sync(self, session, action):
    done = False
    observation_ = None
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text_sync(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    return observation, reward, done
  
def get_file_names(folder_path):
    """
    Gets all file names in a folder.
    """
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names

class LogParser():
  def __init__(self):
    self.actions = []
  
  def get_data(self, file, task, method, **kwargs):
    if task =="gameof24":
      if method == "tot":
        return self.get_tot_results_gameof24(file)
      elif method == "foa":
        data = self.get_foa_data_gameof24(file)
        data.update({"Batching": not "1batch" in file})
        data.update({"Caching": not "nocache" in file})
        return data
      else:
        raise ValueError("Invalid method for gameof24")

    elif task == "crosswords":
      if method == "tot":
        return self.get_tot_results_crosswords(file)
      elif method == "foa":
        data = self.get_foa_data_crosswords(file)
        data.update({"Batching": not "1batch" in file})
        data.update({"Caching": not "nocache" in file})
        return data
      elif method == "bok":
        assert "num_agents" in kwargs, "'num_agents' must also be provided"
        data = self.get_bok_data_crosswords(file, num_agents=kwargs.get("num_agents"))
        data.update({"Batching": False})
        data.update({"Caching": not "nocache" in file})
        return data

    elif task == "webshop":
      if method == "foa":
        data = self.get_foa_data_webshop(file)
        data.update({"Batching": not "1batch" in file})
        data.update({"Caching": not "nocache" in file})
        return data
      elif method == "react":
        assert "num_agents" in kwargs, "'num_agents' must also be provided"
        data = self.get_react_data_webshop(file, num_agents=kwargs.get("num_agents"))
        data.update({"Batching": False})
        data.update({"Caching": not "nocache" in file})
        return data
      else:
        raise NotImplementedError("Webshop method not implemented yet")
    
    else:
      raise ValueError("Invalid task")
    
  def get_tot_results_gameof24(self, file_name):
    with open(file_name, 'r') as file:
      run = json.load(file)

    # Compute cost
    cost = run[-1]["usage_so_far"]["cost"]

    # Compute accuracy
    rewards = []
    for puzzle in run:
      rewards.append({"r":1} in puzzle["infos"])
    
    return rewards, cost
  
  def get_foa_data_gameof24(self, file):
    with open(file, 'r') as f:
      run = json.load(f)
    
    data = {}
    params = file.split("/")[-1].split("_")
    data["num_agents"] = int(params[1].split("agents")[0])
    data["num_steps"] = int(params[2].split("steps")[0])
    data["k"] = int(params[3].split("k")[0])
    data["backtrack"] = float(params[5].split("backtrack")[0])

    if "linear_filtered-resampling" in file:
      data["resampling"] = "linear_filtered"
    elif "linear-resampling" in file:
      data["resampling"] = "linear"
    else:
      raise ValueError("No resampling method found in the log file's name")

    if "Info" in run.keys():
      data["Cost"] = run.pop("Info")["Cost"]["Total cost"]["total_cost"]
    elif "Cost" in run.keys():
      data["Cost"] = run.pop("Cost")["total_cost"]
    else:
      raise ValueError("No cost found in the log file")

    rewards = []
    for puzzle in run.values():
      rewards.append(int({"r":1} in puzzle["Verifications"]))
    data["rewards"] = rewards
    stats = get_stats(np.array(rewards), "Accuracy")
    data.update(stats)
    return data

  def get_tot_results_crosswords(self, file_path):

    with open(file_path) as f:
      results = json.load(f)

    cost = results.pop(-1)["cost"]
    best_steps = []
    for game in results:
      step_len = [len(step["actions"]) for step in game]
      if step_len == []:
        # Empty game -> No suggestions at root node
        best_steps.append({"total_step":0, "env_step":0, "actions":[], 'info': {'r_letter': 0, 'r_word': 0},})
        continue
      best_step_index = step_len.index(max(step_len))
      best_step = game[best_step_index]
      best_steps.append(best_step)
  
    r_letters = [game["info"]["r_letter"] for game in best_steps]
    r_words = [game["info"]["r_word"] for game in best_steps]
    r_game = [1 if game["info"]["r_word"]==1 else 0 for game in best_steps]

    rewards = {"r_letter": r_letters, "r_word": r_words, "r_game": r_game}
    return rewards, cost
  
  def get_foa_data_crosswords(self, file, metric="r_letter"):
    with open(file, "r") as experiment_file:
      run = json.load(experiment_file)
    
    data = {}
    params = file.split("/")[-1].split("_")
    data["num_agents"] = int(params[1].split("agents")[0])
    data["num_steps"] = int(params[2].split("steps")[0])
    data["k"] = int(params[3].split("k")[0])
    data["backtrack"] = float(params[5].split("backtrack")[0])

    if "linear_filtered-resampling" in file:
      data["resampling"] = "linear_filtered"
    elif "linear-resampling" in file:
      data["resampling"] = "linear"
    else:
      raise ValueError("No resampling method found in the log file's name")

    if "Info" in run.keys():
      data["Cost"] = run.pop("Info")["Cost"]["Total cost"]["total_cost"]
    elif "Cost" in run.keys():
      data["Cost"] = run.pop("Cost")['Total cost']["total_cost"]
    else:
      raise ValueError("No cost found in the log file")

    metrics = {}
    for puzzle_id, puzzle_results in run.items():
      initial_puzzle = puzzle_results.pop("puzzle", None)       # Not needed just want to pop
      verifications = puzzle_results.pop("Verifications", None) # Not needed just want to pop

      max_actions = 0
      metrics[puzzle_id] = {"r_letter": None, "r_word": None, "r_all": None}
      for agent_id, agent_results in puzzle_results.items():
        for step_id, step_results in agent_results.items():
          step_actions = len(step_results["Step"].split(" -> "))
          if step_actions > max_actions:
            max_actions = step_actions
            metrics[puzzle_id] = step_results["metrics"]
      assert max_actions > 0, f"No actions found for {puzzle_id}"

    r_letters = [metric["r_letter"] for metric in metrics.values()]
    r_words = [metric["r_word"] for metric in metrics.values()]
    r_alls = [metric["r_all"] for metric in metrics.values()]
    metrics = {"r_letter": r_letters, "r_word": r_words, "r_all": r_alls}

    rewards = np.array(metrics[metric])
    data["rewards"] = rewards.tolist()
    #data.update(metrics)
    stats = get_stats(rewards, "Accuracy")
    data.update(stats)
    
    return data
  
  def get_bok_data_crosswords(self, folder, num_agents=1):
    data = {}

    files = [folder + f for f in get_file_names(folder)]
    assert num_agents <= len(files), f"Number of max iid agents : {len(files)}"
    files = files[:num_agents]

    total_cost = 0
    rewards = []

    num_steps = None

    for file in files:
      agent_data = self.get_foa_data_crosswords(file)
      rewards.append(agent_data["rewards"])
      total_cost += agent_data["Cost"]
      
      if num_steps:
        assert num_steps == agent_data["num_steps"], "Number of steps do not match"
      num_steps = agent_data["num_steps"]

    data["num_agents"] = num_agents
    data["num_steps"] = num_steps
    data["k"] = 0
    data["backtrack"] = 0
    data["resampling"] = None
    data["Cost"] = total_cost

    rewards = np.array(rewards).max(axis=0)
    data["rewards"] = rewards.tolist()
    data["success_rate"] = np.mean(rewards == 1)    
    stats = get_stats(rewards, "Accuracy")
    data.update(stats)

    return data 

     
  def get_react_data_webshop(self, folder, num_agents=1):
    data = {}
    
    files = [folder + f for f in get_file_names(folder)]
    assert num_agents <= len(files), f"Number of max iid agents : {len(files)}"
    files = files[:num_agents]

    total_cost = 0
    rewards = []

    num_steps = None
    prompting = None
    
    for file in files:
      agent_data = self.get_foa_data_webshop(file)
      rewards.append(agent_data["rewards"])
      total_cost += agent_data["Cost"]
      
      if num_steps:
        assert num_steps == agent_data["num_steps"], "Number of steps do not match"
      num_steps = agent_data["num_steps"]

      if prompting:
        assert prompting == agent_data["prompting"], "Prompting do not match"
      prompting = agent_data["prompting"]
    
    data["num_agents"] = num_agents
    data["num_steps"] = num_steps
    data["k"] = 0
    data["backtrack"] = 0
    data["prompting"] = prompting
    data["resampling"] = None
    data["Cost"] = total_cost

    rewards = np.array(rewards).max(axis=0)
    data["rewards"] = rewards.tolist()
    data["success_rate"] = np.mean(rewards == 1)    
    stats = get_stats(rewards, "Accuracy")
    data.update(stats)

    return data
  def get_foa_data_webshop(self, file):
    with open(file) as f:
      run = json.load(f)
    info = run.pop("Info")

    data = {}
    data["num_agents"] = info["FoA options"]["num_agents"]
    data["num_steps"] = info["FoA options"]["num_steps"]
    data["k"] = info["FoA options"]["k"]
    data["backtrack"] = info["FoA options"]["backtrack"]
    data["prompting"] = info["FoA options"]["prompting"]
    data["Cost"] = info["Cost"]["Total cost"]["total_cost"]
    data["resampling"] = info["FoA options"]["resampling_method"]
    
    rewards = []
    for puzzle in run.values():
      environment = puzzle.pop("environment")
      agent_rewards = []
      for agent in puzzle.values():
        last_step = list(agent.keys())[-1]
        reward = agent[last_step]["Latest reward"]
        agent_rewards.append(reward)
      rewards.append(agent_rewards)

    rewards = np.array(rewards)
    rewards = rewards.max(axis=1)

    assert np.mean(rewards) == info["Metrics"]["mean_reward"], "Mean reward does not match: Expected {}, Got {}".format(info["Metrics"]["mean_reward"], np.mean(rewards))
    assert np.mean(rewards>0) == info["Metrics"]["percentage_finished"], "Finished agents (%) does not match"

    
    data["rewards"] = rewards.tolist()
    data["success_rate"] = np.mean(rewards==1)
    stats = get_stats(rewards, "Accuracy")
    data.update(stats)

    return data
  

  

def get_stats(array, name):
   bootstrapped = bootstrap(array.reshape((1,-1)), np.mean)
   mean = bootstrapped.bootstrap_distribution.mean()
   ci = bootstrapped.confidence_interval
   stats = {name: mean, "ci_low": ci[0], "ci_high": ci[1], "error_low": mean - ci[0], "error_high": ci[1] - mean}
   return stats


def merge_responses(responses):
   # Given a list of OpenAI/Groq styled responses, merge them into a single response
   # The responses are assumed to be in the same format and of the same input prompt

    # Merge the responses
    merged_response = responses[0]
    prompt_usage = merged_response.usage.prompt_tokens

    for i, response in enumerate(responses[1:], start=1):
        
        merged_response.choices.extend(response.choices)        
        
        assert prompt_usage == response.usage.prompt_tokens, f"Prompt tokens do not match for response {i}"
        
        
        # Update the usage
        merged_response.usage.completion_tokens += response.usage.completion_tokens
        merged_response.usage.prompt_tokens     += 0                                  # Count prompt tokeny only once
        merged_response.usage.total_tokens      += response.usage.completion_tokens   # Count prompt tokeny only once
        
        # Only for Groq (not for TogetherAI)
        try:
            merged_response.usage.completion_time   += response.usage.completion_time
            merged_response.usage.prompt_time       += 0                                  # Count prompt tokeny only once
            merged_response.usage.total_time        += response.usage.completion_time     # Count prompt tokeny only once
            merged_response.usage.queue_time        += response.usage.queue_time
        except AttributeError as e:
           pass


    # Set the correct index for each choice
    for i, choice in enumerate(merged_response.choices):
        choice.index = i

    return merged_response      

def load_test_puzzles():
  with open("test_puzzles2.pkl", "rb") as f:
      puzzles = pickle.load(f)
  return puzzles  
