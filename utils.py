import os
import smtplib
import json

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


def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def webshop_text(webshop_url, session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{webshop_url}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{webshop_url}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{webshop_url}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{webshop_url}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{webshop_url}/done/{session}/'
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
        return clean_str(observation), info
    
def get_file_names(folder_path):
    """
    Gets all file names in a folder.
    """
    file_names = []
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            file_names.append(file_name)
    return file_names