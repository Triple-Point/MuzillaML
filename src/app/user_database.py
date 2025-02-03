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
        """
        Args:
            file_path (str): Path to the JSON file.
        """
        self.file_path = file_path
        self.users = defaultdict(default_user)

    def read_json(self) -> bool:
        """
        Reads the JSON file and returns the data as a Python dictionary.

        Returns:
            bool: True if the data was read successfully, False otherwise.
        """
        try:
            with open(self.file_path, 'r') as file:
                self.users = json.load(file)
                return True
        except FileNotFoundError:
            print(f"File not found: {self.file_path}. Returning an empty dictionary.")
            return False
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}. Returning an empty dictionary.")
            return False

    def write_json(self) -> bool:
        """
        Writes a Python dictionary to a JSON file.
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
