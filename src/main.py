import argparse

import joblib
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.model_selection import KFold

from data.DataUtils import load_data, split_user
from src.models.tfidf_model import TFIDFModel


def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")




def get_similar_user_index(set1, train_matrix):
    cosine_sim = cosine_similarity(set1, train_matrix).flatten()
    # Find the index of the most similar user
    most_similar_user_index = np.argmax(cosine_sim)
    #print(f"Cosine Similarities {cosine_sim.shape}:\n{cosine_sim}")
    #print(f"Most Similar User Index {most_similar_user_index} with value {cosine_sim[most_similar_user_index]}")
    return most_similar_user_index, cosine_sim[most_similar_user_index]


def main(config_file):
    config = load_config(config_file)

    # If no processed data, create it
    user_artist_matrix = load_data(config['data']['raw_data'])

    # 10-fold Cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(user_artist_matrix):
        train_matrix = user_artist_matrix[train_index]
        test_matrix = user_artist_matrix[test_index]

        total_score = 0

        # For each user in the test set
        for user in test_matrix:
            set1, set2 = split_user(user)

            # Current assumption for KPI is that if the similarity is lower with full user data, that's a win
            peer, score = get_similar_user_index(set1, train_matrix)
            target_score = cosine_similarity(user, train_matrix[peer]).flatten()
            total_score += (target_score - score)

            peer, score = get_similar_user_index(set2, train_matrix)
            target_score = cosine_similarity(user, train_matrix[peer]).flatten()
            total_score += (target_score - score)

        print(f"{total_score=}")

    tfidf_model = TFIDFModel(user_artist_matrix)
    recommended_docs = tfidf_model.get_recommendations(0)
    print(recommended_docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
