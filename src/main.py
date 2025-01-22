import argparse
import logging

import yaml
from data.DataUtils import normalize_sparse_tensor, load_or_create, cross_validate
from src.data.DataUtils import dump_to_file
from src.models.cosine_model import CosineModel
from src.models.tfidf_model import TFIDFModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    if 'image_dump' in config['data'] and config['data']['image_dump']:
        dump_to_file(user_artist_matrix, config['data']['image_dump'])

    # TODO: Factory?
    model_type = config['model']['model']
    if model_type == 'cosine_model':
        model_class = CosineModel
    elif model_type == 'tfidf_model':
        model_class = TFIDFModel
    else:
        raise ValueError("model must be defined in configuration file:\nmodel:\n\tmodel: [cosine_model|tfidf_model]")

    eval_scores = []
    for train_tensor, test_tensor in cross_validate(user_artist_matrix):
        # Make a model on the train data
        model = model_class(train_tensor)
        eval_scores.append(model.evaluate(test_tensor))

    print(f"{eval_scores=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
