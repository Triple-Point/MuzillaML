import yaml
from flask import Flask, request, render_template, redirect, url_for
import sqlite3
import random
import argparse

from src.data.DataUtils import load_or_create
from src.models.cosine_model import CosineModel

app = Flask(__name__)

model = None

def get_db_connection(database='database.db'):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    return connection

def create_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        liked_artists TEXT,
        disliked_artists TEXT
    )''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form.get('user_id')
    if not user_id:
        user_id = str(random.randint(1000, 9999))

    # Insert or get user
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    if not user:
        conn.execute('INSERT INTO users (id, liked_artists, disliked_artists) VALUES (?, ?, ?)',
                     (user_id, '', ''))
        conn.commit()
    conn.close()

    return redirect(url_for('recommend', user_id=user_id))

@app.route('/recommend/<user_id>', methods=['GET', 'POST'])
def recommend(user_id):
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    liked_artists = user['liked_artists'].split(',')
    disliked_artists = user['disliked_artists'].split(',')

    # Call your recommendation model to get a list of new artists
    new_recommendations = model_recommend_items(liked_artists, )[:10]

    if request.method == 'POST':
        artist = request.form['artist']
        action = request.form['action']
        if action == 'like':
            liked_artists.append(artist)
        elif action == 'dislike':
            disliked_artists.append(artist)

        # Update user preferences
        conn.execute('UPDATE users SET liked_artists = ?, disliked_artists = ? WHERE id = ?',
                     (','.join(liked_artists), ','.join(disliked_artists), user_id))
        conn.commit()

        new_recommendations = model_recommend_items(user_id)[:10]

    conn.close()
    return render_template('recommend.html', user_id=user_id, recommendations=new_recommendations)

def model_recommend_items(user, num):
    # Dummy function for recommendation
    recommended_artists = model.recommend_items_list(user, num)
    artists = ['Artist_'+str(i) for i in recommended_artists]
    return random.sample(artists, 10)

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
    create_database()
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    user_artist_matrix, user_lookup, artist_lookup = load_or_create(config['data']['raw_data'], config['data']['processed_data'])
    model = CosineModel(user_artist_matrix)
    app.run(debug=True)
