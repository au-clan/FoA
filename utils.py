import os
import smtplib
import json
import asyncio
import aiohttp

import numpy as np
from pathlib import Path

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


def email_notification(subject: str, message: str, reciever_email: str="nearchospot@gmail.com"):

    sender_email = "nearchospot@gmail.com"

    text = f"Subject: {subject}\n\n{message}"

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    try:
        google_app_pass = os.environ.get("GOOGLE_APP_PASS")
    except:
        raise EnvironmentError("Environment variable <GOOGLE_APP_PASS> not found")
    
    google_app_pass = os.environ.get("GOOGLE_APP_PASS")
    server.login(sender_email, google_app_pass)

    server.sendmail(sender_email, reciever_email, text)
    return

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
    
    api_cost = api.cost(actual_cost=True)["total_cost"]

    new_cost = current_cost + api_cost

    with open('actual_cost.txt', 'w') as file:
        file.write(str(new_cost))


WEBSHOP_URL = "http://10.90.39.11:3000"
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