import argparse

import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from data.DataUtils import load_data, split_user, csr_to_torch_sparse_tensor, normalize_sparse_tensor
from src.models.cosine_model import sparse_cosine_similarity
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



def get_similar_user_index(new_user, user_tensor):
    # Compute cosine similarity
    cosine_sim = sparse_cosine_similarity(new_user, user_tensor)

    # Convert result back to dense for inspection if needed
    cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()
    #print(cosine_sim.cpu().numpy())

    # Find the index of the most similar user
    most_similar_user_index = np.argmax(cosine_sim)
    similarity_value = cosine_sim[most_similar_user_index]
    #print(f"Cosine Similarities {cosine_sim.shape}:\n{cosine_sim}")
    print(f"{most_similar_user_index=}\t{similarity_value=}")
    return most_similar_user_index, similarity_value


def main(config_file):
    config = load_config(config_file)

    # If no processed data, create it
    user_artist_matrix = load_data(config['data']['raw_data'])

    # 10-fold Cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(user_artist_matrix):
        train_matrix = user_artist_matrix[train_index]
        test_matrix = user_artist_matrix[test_index]

        train_tensor = user_artist_matrix[train_index]

        # Convert to PyTorch sparse tensors and move to GPU
        train_tensor = csr_to_torch_sparse_tensor(train_tensor).to('cuda')
        train_tensor = normalize_sparse_tensor(train_tensor)

        total_score = 0

        # For each user in the test set
        for user in test_matrix:
            set1, set2 = split_user(user)

            set1_tensor = normalize_sparse_tensor(csr_to_torch_sparse_tensor(set1).to('cuda'))

            # Current assumption for KPI is that if the similarity is lower with full user data, that's a win
            peer, score = get_similar_user_index(set1_tensor, train_tensor)
            target_score = cosine_similarity(user, train_matrix[peer]).flatten()
            total_score += (target_score - score)

            set2_tensor = normalize_sparse_tensor(csr_to_torch_sparse_tensor(set2).to('cuda'))

            peer, score = get_similar_user_index(set2_tensor, train_tensor)
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
