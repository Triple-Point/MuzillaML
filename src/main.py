import argparse
import logging
import yaml

from data.DataUtils import load_or_create, remove_random_values
from src.data.DataUtils import dump_to_image
from src.data.TensorUtils import distribute_sparse_tensor, concatenate_except_one, get_users
from src.metrics.AveragePrecision import average_precision

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'cosine_model': 'src.models.cosine_model.CosineModel',
    'random_model': 'src.models.random_model.RandomModel',
    'tfidf_model': 'src.models.tfidf_model.TFIDFModel',
    'popularity_model': 'src.models.popularity_model.PopularityModel'
}


def load_config(config_file):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def evaluate(model, test_tensor):
    """
    Evaluate the model using mean average precision.

    Args:
        model: The trained model to evaluate.
        test_tensor: Sparse tensor for test data.

    Returns:
        float: Mean average precision score for the evaluation.
    """
    total_score = 0
    i = 0
    for i, user in enumerate(get_users(test_tensor)):
        # TODO: Batch some of this to save time?
        masked_user, masked_artists = remove_random_values(user)
        # Generate the top-n artists
        new_artist_list = model.recommend_items(masked_user, len(masked_artists))
        total_score += average_precision(new_artist_list, masked_artists)
        average_score = total_score / (i + 1)
        logger.info(f'{i=}\t{total_score=}\t{average_score=}')
    logger.info(f"Tested users {i} in the set of size {test_tensor.size()}")
    return total_score / (i + 1)


def main(config_file):
    config = load_config(config_file)

    # If no processed data, create it
    user_artist_matrix, user_lookup, artist_lookup = load_or_create(config['data']['raw_data'], config['data']['processed_data'])

    # Dump matrix to a PNG file - TODO: Requires a dense representation, so memory issues here
    if 'image_dump' in config['data'] and config['data']['image_dump']:
        dump_to_image(user_artist_matrix, config['data']['image_dump'])

    try:
        model_type = config['model']['model']
    except KeyError:
        logger.error("model must be defined in configuration file:\nmodel:\n\tmodel: ["
                     f"{'|'.join([k for k in MODEL_CLASSES])}]")
        exit()

    model_path = MODEL_CLASSES.get(model_type)
    if not model_path:
        raise ValueError(f"Invalid model type: {model_type}")

    module_path, class_name = model_path.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    try:
        num_folds = config['model']['cross_validation_folds']
    except KeyError:
        num_folds = 10

    eval_scores = []
    logger.info(f"Splitting into {num_folds} cross-validation sets")
    user_buckets = distribute_sparse_tensor(user_artist_matrix, num_folds)

    # Check: Make and test a model on a single fold of the data
    model = model_class(user_buckets[0])
    print(evaluate(model, user_buckets[0]))

    for i, test_tensor in enumerate(user_buckets):
        logger.info(f"Starting a cross-validation batch {i}")
        train_tensor = concatenate_except_one(user_buckets, i)
        # Make a model on the train data
        logger.debug(f"Creating model")
        model = model_class(train_tensor)
        logger.debug(f"About to evaluate")
        score = evaluate(model, test_tensor)
        eval_scores.append(score)
        logger.info(f"Cross-validation results: {eval_scores=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
