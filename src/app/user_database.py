import json
from collections import defaultdict


def default_user():
    return {
        'liked_artists': [],
        'album_counts': [],
        'disliked_artists': []
    }


class UserDatabase:
    def __init__(self, file_path):
        self.file_path = file_path
        self.users = defaultdict(default_user)

    def read_json(self):
        """
        Reads the JSON file and returns the data as a Python dictionary.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: Data from the JSON file.
        """
        try:
            with open(self.file_path, 'r') as file:
                self.users = json.load(file)
        except FileNotFoundError:
            print(f"File not found: {self.file_path}. Returning an empty dictionary.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Returning an empty dictionary.")
            return {}

    def write_json(self):
        """
        Writes a Python dictionary to a JSON file.

        Args:
            file_path (str): Path to the JSON file.
            data (dict): Data to write to the JSON file.

        Returns:
            bool: True if the data was written successfully, False otherwise.
        """
        try:
            with open(self.file_path, 'w') as file:
                json.dump(self.users, file, indent=4)
                return True
        except IOError as e:
            print(f"Error writing to file: {e}")
            return False
