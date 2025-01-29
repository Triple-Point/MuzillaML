import yaml
from flask import Flask, request, render_template, redirect, url_for
import random
import argparse
import json

from src.data.FileUtils import load_or_create, ArtistLookup
from src.models.cosine_model import CosineModel
from src.models.model_factory import model_factory

app = Flask(__name__)

model = None

FILE_PATH = "data/user_db.json"


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
artist_name_lookup = None

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
    global artist_name_lookup

    if request.method == 'POST':
        artist = request.form['artist']
        action = request.form['action']
        if action == 'like':
            user['liked_artists'].extend(artist_name_lookup.artists_to_ids([artist]))
            user['album_count'].append(1)
        elif action == 'dislike':
            user['disliked_artists'].append(artist)

        # Update user preferences
        write_json(FILE_PATH, users)

        new_recommendations = model_recommend_items(user['liked_artists'], user['album_count'], ON_SCREEN_ARTISTS)
    liked_strings = artist_name_lookup.ids_to_artists(user['liked_artists'])
    disliked_strings = artist_name_lookup.ids_to_artists(user['disliked_artists'])

    return render_template('recommend.html', user_id=user_id, recommendations=new_recommendations,
                           liked_artists=liked_strings, disliked_artists=disliked_strings)


def model_recommend_items(artist_list: list[int], album_count: list[int], num: int) -> list[str]:
    print(f"{artist_list=}")
    recommended_artists = model.recommend_items_list(artist_list, album_count, num)
    global artist_name_lookup
    artist_strings = artist_name_lookup.outputs_to_artists(recommended_artists)
    print(f"{artist_strings=}")
    return artist_strings


def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def main(config):
    global model
    global artist_name_lookup
    user_artist_matrix, user_lookup, artist_id_lookup = load_or_create(config['data']['raw_data'],
                                                                       config['data']['processed_data'])
    artist_name_lookup = ArtistLookup(artist_id_lookup, config['data']['artist_lookup_table'])
    model_class = model_factory(config['model']['model'])
    model = model_class(user_artist_matrix)
    app.run(debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
