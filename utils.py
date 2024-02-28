import os
import smtplib

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


def email_notification(subject, message):

    sender_email = "nearchospot@gmail.com"
    reciever_email = "nearchospot@gmail.com"

    text = f"Subject: {subject}\n\n{message}"

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    try:
        google_app_pass = os.environ.get("GOOGLE_APP_PASS")
    except:
        print("Environment variable <GOOGLE_APP_PASS> not found")
        return
    
    google_app_pass = os.environ.get("GOOGLE_APP_PASS")
    server.login(sender_email, google_app_pass)

    server.sendmail(sender_email, reciever_email, text)
    print("Email sent successfully")
    return