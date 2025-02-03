import yaml
from flask import Flask, request, render_template, redirect, url_for
import random
import argparse

from src.app.user_database import UserDatabase
from src.app.web_recommender import WebRecommender

app = Flask(__name__)

model: WebRecommender
user_db: UserDatabase


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
    user = user_db.users[user_id]

    # Call your recommendation model to get a list of new artists
    new_recommendations = model.recommend_items(user['liked_artists'], user['album_counts'], ON_SCREEN_ARTISTS,
                                                user['disliked_artists'])

    if request.method == 'POST':
        artist = request.form['artist']
        action = request.form['action']
        artist_id = model.artist_name_lookup.artists_to_ids([artist])[0]
        if action == 'like':
            user['liked_artists'].append(artist_id)
            user['album_counts'].append(1)
        elif action == 'dislike':
            user['disliked_artists'].append(artist_id)
        elif action == 'increment':
            i = user['liked_artists'].index(artist_id)
            user['album_counts'][i] += 1
        elif action == 'decrement':
            i = user['liked_artists'].index(artist_id)
            if user['album_counts'][i] == 1:
                del user['album_counts'][i]
                del user['liked_artists'][i]
            else:
                user['album_counts'][i] -= 1
        elif action == 'delete':
            user['disliked_artists'].remove(artist_id)

        # Update user preferences
        user_db.write_json()

        new_recommendations = model.recommend_items(user['liked_artists'], user['album_counts'], ON_SCREEN_ARTISTS,
                                                    user['disliked_artists'])
    liked_strings = model.artist_name_lookup.ids_to_artists(user['liked_artists'])
    disliked_strings = model.artist_name_lookup.ids_to_artists(user['disliked_artists'])

    return render_template('recommend.html', user_id=user_id, recommendations=new_recommendations,
                           liked_artists=liked_strings, album_counts=user['album_counts'],
                           disliked_artists=disliked_strings)


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
    model = WebRecommender(config['data']['artist_lookup_table'], config['model']['model'], config['data']['raw_data'],
                           config['data']['processed_data'])
    global user_db
    user_db = UserDatabase(config['data']['app_users'])
    app.run(debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    main(config)
