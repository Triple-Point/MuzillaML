import argparse
import logging
import yaml

from src.data.FileUtils import dump_to_image
from src.models.model_cross_validator import CrossValidator
from src.models.model_factory import MODEL_CLASSES

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

    try:
        model_type = config['model']['model']
    except KeyError:
        logger.error("model must be defined in configuration file:\nmodel:\n\tmodel: ["
                     f"{'|'.join([k for k in MODEL_CLASSES])}]")
        exit()

    try:
        num_folds = config['model']['cross_validation_folds']
    except KeyError:
        num_folds = 10

    cross_validator = CrossValidator.from_files(config['data']['raw_data'], config['data']['processed_data'])
    cross_validator.set_cross_validation_params(model_type, num_folds)

    # Dump matrix to a PNG file - TODO: Requires a dense representation, so memory issues here
    if 'image_dump' in config['data'] and config['data']['image_dump']:
        dump_to_image(cross_validator.data, config['data']['image_dump'])

    cross_validation_results = cross_validator.evaluate()
    logger.info(f"Cross-validation results: {cross_validation_results=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
