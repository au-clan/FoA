import os, shutil
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