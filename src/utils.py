import os, shutil
from pathlib import Path


def create_folder(folder_path: str):
    """
    Creates a folder if it doesn't exist
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)

def empty_folder(folder: str):
    """
    Deletes all the contents of a folder
    """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def delete_file(file_path: str):
    """
    Deletes a file given its path, if it already exists
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted : '{file_path}'")