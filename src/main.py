import argparse

import yaml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold

from data.DataUtils import split_user, csr_to_torch_sparse_tensor, normalize_sparse_tensor, load_or_create
from src.data.DataUtils import dump_to_file
from src.models.cosine_model import CosineModel
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


def main(config_file):
    config = load_config(config_file)

    # If no processed data, create it
    user_artist_matrix = load_or_create(config['data']['raw_data'], config['data']['processed_data'])

    # Dump matrix to a PNG file
    if config['data']['image_dump']:
        dump_to_file(user_artist_matrix, config['data']['image_dump'])

    # TODO: Factory?
    model_type = config['model']['model']
    if model_type == 'cosine_model':
        model_class = CosineModel
    elif model_type == 'tfidf_model':
        model_class = TFIDFModel
    else:
        raise ValueError("model must be defined in configuration file:\nmodel:\n\tmodel: [cosine_model|tfidf_model]")

    # 10-fold Cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(user_artist_matrix):
        train_matrix = user_artist_matrix[train_index]
        test_matrix = user_artist_matrix[test_index]

        train_tensor = user_artist_matrix[train_index]

        # Convert to PyTorch sparse tensors and move to GPU
        train_tensor = csr_to_torch_sparse_tensor(train_tensor).to('cuda')
        train_tensor = normalize_sparse_tensor(train_tensor)

        # Make a model on the train data
        model = model_class(train_tensor)

        total_score = 0

        # For each user in the test set
        for user in test_matrix:
            # TODO: Batch some of this to save time
            set1, set2 = split_user(user)

            set1_tensor = normalize_sparse_tensor(csr_to_torch_sparse_tensor(set1).to('cuda'))

            # Current assumption for KPI is that if the similarity is lower with full user data, that's a win
            peer, score = model.get_similar_user_index(set1_tensor)
            target_score = cosine_similarity(user, train_matrix[peer]).flatten()
            total_score += (target_score - score)

            set2_tensor = normalize_sparse_tensor(csr_to_torch_sparse_tensor(set2).to('cuda'))

            peer, score = model.get_similar_user_index(set2_tensor)
            target_score = cosine_similarity(user, train_matrix[peer]).flatten()
            total_score += (target_score - score)

        print(f"{total_score=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
