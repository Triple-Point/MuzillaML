import argparse
import logging

import yaml
from data.DataUtils import normalize_sparse_tensor, load_or_create, cross_validate, remove_random_values
from src.data.DataUtils import dump_to_file
from src.metrics.MeanAveragePrecision import mean_average_precision
from src.models.cosine_model import CosineModel
from src.models.popularity_model import PopularityModel
from src.models.random_model import RandomModel
from src.models.tfidf_model import TFIDFModel
# TODO: all the torch stuff needs to go somewhere else
import torch

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

def evaluate(model, test_tensor):
    total_score = 0
    i = 0
    for i, user in enumerate(get_users(test_tensor)):
        # TODO: Batch some of this to save time
        masked_user, masked_artists = remove_random_values(user)
        # Generate the top-n artists
        new_artist_list = model.recommend_items(masked_user, len(masked_artists))
        total_score += mean_average_precision(new_artist_list, masked_artists)
        average_score = total_score / (i+1)
        logger.info(f'{i=}\t{total_score=}\t{average_score=}')
    print(f"Tested users {i} in the set of size {test_tensor.size()}")
    return total_score/(i+1)


def distribute_sparse_tensor(sparse_tensor, num_buckets=10):
    """
    Distribute the users from a collated torch.sparse_coo_tensor into `num_buckets` tensors cyclically.

    Args:
        sparse_tensor (torch.sparse_coo_tensor): Input sparse tensor in COO format.
        num_buckets (int): Number of output tensors (buckets).

    Returns:
        list: A list of torch.sparse_coo_tensor, one for each bucket.
    """
    sub_tensors = []

    for mod_id in range(num_buckets):
        logger.info(f"Masking for tensor {mod_id}")

        # Get mask for indices[0] % num_sub_tensors == mod_id
        mask = (sparse_tensor.indices()[0] % num_buckets) == mod_id

        # Slice indices and values
        sub_indices = sparse_tensor.indices()[:, mask]
        sub_values = sparse_tensor.values()[mask]

        # Create sub-tensor
        sub_tensors.append(torch.sparse_coo_tensor(
            sub_indices, sub_values, sparse_tensor.size(), device="cuda"
        ).coalesce())
        logger.info(f"Tensor {mod_id} contains {sub_tensors[-1].indices().shape[1]} non-zero values")
    return sub_tensors


def concatenate_except_one(sub_tensors, excluded_index):
    """
    Concatenate all tensors except the one at the excluded index.

    Args:
        sub_tensors (list of torch.sparse_coo_tensor): List of sparse tensors.
        excluded_index (int): Index of the tensor to exclude.

    Returns:
        torch.sparse_coo_tensor: The concatenated sparse tensor.
    """
    indices_list = []
    values_list = []
    size = sub_tensors[0].size()  # Assuming all tensors have the same size
    total_rows = 0

    for i, tensor in enumerate(sub_tensors):
        if i == excluded_index:
            continue
        if tensor._nnz() == 0:  # Skip empty tensors
            continue

        # Offset row indices by the current total_rows
        indices = tensor.indices()
        indices[0] += total_rows
        indices_list.append(indices)

        values_list.append(tensor.values())

        # Update total rows
        total_rows += tensor.size(0)

    # Concatenate indices and values
    if indices_list:
        concatenated_indices = torch.cat(indices_list, dim=1)
        concatenated_values = torch.cat(values_list)
    else:
        concatenated_indices = torch.empty((2, 0), dtype=torch.int64)
        concatenated_values = torch.empty(0, dtype=torch.float32)

    # Create the concatenated sparse tensor
    concatenated_size = (total_rows, size[1])
    concatenated_tensor = torch.sparse_coo_tensor(concatenated_indices, concatenated_values, size=concatenated_size)

    return concatenated_tensor.coalesce()

def get_users(sparse_tensor):
    # Iterate over each unique user ID
    unique_user_ids = torch.unique(sparse_tensor.indices()[0])

    for user_id in unique_user_ids:
        # Get mask for current user ID
        mask = sparse_tensor.indices()[0] == user_id

        # Slice the indices and values for the current user ID
        user_indices = sparse_tensor.indices()[:, mask]
        user_values = sparse_tensor.values()[mask]

        # Adjust the row index (user-specific rows start from 0 for dense tensors)
        user_dense = torch.zeros((1, sparse_tensor.size(1)), device="cuda")
        user_dense[0, user_indices[1]] = user_values

        yield user_dense



def main(config_file):
    config = load_config(config_file)

    # If no processed data, create it
    user_artist_matrix = load_or_create(config['data']['raw_data'], config['data']['processed_data']).to("cuda")

    # Dump matrix to a PNG file
    if 'image_dump' in config['data'] and config['data']['image_dump']:
        dump_to_file(user_artist_matrix, config['data']['image_dump'])

    # TODO: Factory?
    model_type = config['model']['model']
    if model_type == 'cosine_model':
        model_class = CosineModel
    elif model_type == 'random_model':
        model_class = RandomModel
    elif model_type == 'tfidf_model':
        model_class = TFIDFModel
    elif model_type == 'popularity_model':
        model_class = PopularityModel
    else:
        raise ValueError("model must be defined in configuration file:\nmodel:\n\tmodel: [cosine_model|tfidf_model|popularity_model|random_model]")

    eval_scores = []
    XVALID_FOLDS = 10
    logger.info(f"Splitting into {XVALID_FOLDS} cross-validation sets")
    user_buckets = distribute_sparse_tensor(user_artist_matrix, XVALID_FOLDS)
    for i, test_tensor in enumerate(user_buckets):
        logger.info(f"Starting a cross-validation batch {i}")
        train_tensor = concatenate_except_one(user_buckets, i)
        # Make a model on the train data
        logger.info(f"Creating model")
        model = model_class(train_tensor)
        # We do not want to use a COO matrix here, since we want to iterate over the users (maybe some matrix multiplication)
        logger.info(f"to torch.sparse_csr_tensor")
        logger.info(f"About to evaluate")
        score = evaluate(model, test_tensor)
        eval_scores.append(score)

        logger.info(f"{eval_scores=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Experiment Configurations')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    main(args.config)
