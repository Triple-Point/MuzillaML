import csv

import yaml
from flask import Flask, request, render_template, redirect, url_for
import random
import argparse
import json

from src.data.FileUtils import load_or_create
from src.models.cosine_model import CosineModel

app = Flask(__name__)

model = None

FILE_PATH = "data/user_db.json"

ARTIST_LOOKUP = "data/artists_id_name.csv"


def read_json(file_path):
    """
    Reads the JSON file and returns the data as a Python dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Data from the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}. Returning an empty dictionary.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}. Returning an empty dictionary.")
        return {}


def write_json(file_path, data):
    """
    Writes a Python dictionary to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (dict): Data to write to the JSON file.

    Returns:
        bool: True if the data was written successfully, False otherwise.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            return True
    except IOError as e:
        print(f"Error writing to file: {e}")
        return False


users = read_json(FILE_PATH)


def load_csv_to_dict(file_path):
    """
    Loads a CSV file containing key-value pairs into a dictionary.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary with keys and values from the CSV file.
    """
    result_dict = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:  # Ensure there are exactly two columns
                    key, value = row
                    result_dict[int(key)] = value
                else:
                    print(f"Skipping invalid row: {row}")
        return result_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


artist_lookup = load_csv_to_dict(ARTIST_LOOKUP)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('user_id')
    if not user_id:
        user_id = "User_" + str(random.randint(1000, 9999))

    # Insert or get user
    return redirect(url_for('recommend', user_id=user_id))


ON_SCREEN_ARTISTS = 8


@app.route('/recommend/<user_id>', methods=['GET', 'POST'])
def recommend(user_id):
    # TODO: Default dict?
    if user_id not in users:
        users[user_id] = {
            'liked_artists': [],
            'album_count': [],
            'disliked_artists': []
        }
    user = users[user_id]

    # Call your recommendation model to get a list of new artists
    new_recommendations = model_recommend_items(user['liked_artists'], user['album_count'], ON_SCREEN_ARTISTS)
    print(f"{new_recommendations=}")

    if request.method == 'POST':
        artist = request.form['artist']
        action = request.form['action']
        if action == 'like':
            user['liked_artists'].append(artist)
            user['album_count'].append(1)
        elif action == 'dislike':
            user['disliked_artists'].append(artist)

        # Update user preferences
        write_json(FILE_PATH, users)

        new_recommendations = model_recommend_items(user['liked_artists'], user['album_count'], ON_SCREEN_ARTISTS)

    return render_template('recommend.html', user_id=user_id, recommendations=new_recommendations)


def model_recommend_items(artist_list, album_count, num):
    # Dummy function for recommendation
    print(artist_list, album_count, num)
    recommended_artists = model.recommend_items_list(artist_list, album_count, num)
    print(f"{recommended_artists=}")
    print(f"{artist_lookup=}")
    artists = [artist_lookup[i] if i in artist_lookup else "<UNKNOWN_ARTIST>" for i in recommended_artists]
    return artists


def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    user_artist_matrix, user_lookup, artist_lookup_ = load_or_create(config['data']['raw_data'],
                                                                    config['data']['processed_data'])
    model = CosineModel(user_artist_matrix)
    app.run(debug=True)
