import os
import smtplib
import json

import numpy as np
from pathlib import Path


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

    new_suggestions = remove_after_last_bracket(suggestions)
    #valid_suggestions = [suggestion for suggestion in new_suggestions.split("\n") if "(left:" in suggestion]
    return new_suggestions.split("\n")