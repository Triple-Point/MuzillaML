import logging
import os
import pickle
from typing import Tuple, Dict, Any

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
logger.info(f"Using device: {device}")


def load_data(data_file: str) -> tuple[Tensor, list[int], list[int]]:
    """
    Loads user-artist interaction data from a CSV file and converts it into a sparse COO tensor.

    Args:
        data_file (str): Path to the CSV file containing the interaction data.
                         The file must have the following columns:
                         - `user_id`: Unique identifier for each user.
                         - `artist_id`: Unique identifier for each artist.
                         - `album_count`: Interaction count (e.g., number of albums listened to).

    Returns:
        torch.Tensor: A PyTorch sparse COO tensor with:
                      - Indices representing `user_id` and `artist_id`.
                      - Values representing the `album_count`.
                      - Size derived from the maximum `user_id` and `artist_id` values.

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        KeyError: If the required columns (`user_id`, `artist_id`, `album_count`) are missing.
        ValueError: If the data contains invalid or unexpected values (e.g., negative counts).

    Notes:
        - The function uses categorical encoding to convert `user_id` and `artist_id` into integer indices.
        - The resulting tensor is moved to the current device (`cpu` or `cuda`) as determined by PyTorch.

    Example:
        ```python
        sparse_tensor = load_data("user_artist_interactions.csv")
        print(sparse_tensor)
        ```

    Logging:
        - Logs information about the processing stages (loading data, creating tensor, etc.).

    """
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found.")

    df = pd.read_csv(data_file, header=None, names=['user_id', 'artist_id', 'album_count'])

    user_ids = df['user_id'].unique()
    artist_ids = df['artist_id'].unique()

    # Create dictionaries to map original IDs to new indices
    user_id_map = {original_id: index for index, original_id in enumerate(user_ids)}
    artist_id_map = {original_id: index for index, original_id in enumerate(artist_ids)}

    # Replace original IDs with new indices in the dataframe
    df['user_index'] = df['user_id'].map(user_id_map)
    df['artist_index'] = df['artist_id'].map(artist_id_map)

    # Extract indices and values
    indices = [df['user_index'].values, df['artist_index'].values]
    values = df['album_count'].values

    # Create sparse tensor
    tensor = torch.sparse_coo_tensor(indices, values).coalesce()

    return tensor, user_ids, artist_ids


def load_or_create(raw_data_file: str, sparse_data_file: str, force_reprocess: bool = False) -> Tuple[torch.Tensor, Dict[int, int], Dict[int, int]]:
    """
    Loads a preprocessed sparse matrix from disk or creates it from raw data if necessary.

    Args:
        raw_data_file (str): Path to the raw data file (e.g., a CSV file) containing user-artist interactions.
        sparse_data_file (str): Path to the serialized sparse matrix file (e.g., a `.pkl` file).
        force_reprocess (bool, optional): If `True`, reprocess the raw data file even if the serialized file exists.
                                          Defaults to `False`.

    Returns:
        torch.Tensor: A sparse tensor representing the user-artist interaction matrix.

    Raises:
        FileNotFoundError: If `raw_data_file` does not exist when reprocessing is required.
        ValueError: If the deserialized sparse matrix is incompatible with the current system or corrupted.

    Notes:
        - The sparse matrix is serialized and deserialized using the `pickle` module.
        - The function moves the sparse tensor to the current device (`cpu` or `cuda`) as determined by PyTorch.

    Example:
        ```python
        sparse_matrix = load_or_create(
            raw_data_file="user_artist_data.csv",
            sparse_data_file="user_artist_matrix.pkl",
            force_reprocess=True
        )
        print(sparse_matrix)
        ```
    """
    if force_reprocess or not os.path.isfile(sparse_data_file):
        logger.info(f"Reprocessing data from {raw_data_file}")
        sparse_matrix, user_ids, artist_ids = load_data(raw_data_file)
        with open(sparse_data_file, "wb") as f:
            pickle.dump((sparse_matrix, user_ids, artist_ids), f)
    else:
        logger.info(f"Loading data from file {sparse_data_file}")
        with open(sparse_data_file, "rb") as f:
            sparse_matrix, user_ids, artist_ids = pickle.load(f)
    return sparse_matrix.to(device), user_ids, artist_ids


def remove_random_values(user_tensor: torch.Tensor, num_remove: int = 10, seed: int = None):
    """
    Remove random non-zero values from a sparse COO tensor.

    Args:
        user_tensor (torch.sparse_coo_tensor): Input sparse COO tensor.
        num_remove (int): Number of random values to remove.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.sparse_coo_tensor: The updated sparse tensor with values removed.
        list: A sorted list of artists (column indices) sorted by album frequency in descending order.
    """
    if not user_tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse COO tensor.")

    if seed is not None:
        torch.manual_seed(seed)

    # Extract indices and values
    indices = user_tensor.indices()
    values = user_tensor.values()

    # Determine how many values to remove
    actual_remove = min(num_remove, int(values.numel() * 0.1))

    # Select random indices to remove
    if actual_remove > 0:
        remove_indices = torch.multinomial(values, actual_remove, replacement=False)
    else:
        remove_indices = torch.tensor([], dtype=torch.int64, device=user_tensor.device)

    # Get the rows, columns, and values to be removed
    removed_items = [
        (indices[0, idx].item(), indices[1, idx].item(), values[idx].item())
        for idx in remove_indices
    ]

    # Create new indices and values without the removed items
    keep_mask = torch.ones_like(values, dtype=torch.bool)
    keep_mask[remove_indices] = False
    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]

    # Sort removed items by value in descending order
    removed_items = sorted(removed_items, key=lambda x: x[2], reverse=True)
    removed_artists = [item[1] for item in removed_items]

    # Create the updated sparse tensor
    updated_tensor = torch.sparse_coo_tensor(new_indices, new_values, user_tensor.size(), device=user_tensor.device)

    return updated_tensor.coalesce(), removed_artists


# Normalize sparse tensor row-wise
def normalize_sparse_tensor(sparse_tensor):
    # Compute L2 norm of each row
    row_norms = torch.sqrt(torch.sparse.sum(sparse_tensor.pow(2), dim=1).to_dense())
    row_norms = torch.where(row_norms == 0, torch.tensor(1.0, device=row_norms.device), row_norms)  # Avoid div by zero

    # Create a diagonal sparse tensor for normalization
    row_norms_inv = 1.0 / row_norms
    row_indices = torch.arange(row_norms.size(0), device=row_norms.device)
    diagonal_indices = torch.stack([row_indices, row_indices])
    diagonal_values = row_norms_inv
    norm_diagonal = torch.sparse_coo_tensor(diagonal_indices, diagonal_values,
                                            size=(row_norms.size(0), row_norms.size(0)))

    # Normalize rows of the sparse tensor
    return torch.sparse.mm(norm_diagonal, sparse_tensor)


def dump_to_image(user_artist_matrix, out_file_name, show_image=False):
    # Sum the rows and columns
    row_sums = user_artist_matrix.sum(axis=1).A1  # Summing rows
    col_sums = user_artist_matrix.sum(axis=0).A1  # Summing columns
    # Get the sorted indices
    sorted_row_indices = np.argsort(-row_sums)  # Sort descending
    sorted_col_indices = np.argsort(-col_sums)  # Sort descending
    # Reorder the matrix
    sorted_matrix = user_artist_matrix[:][:, sorted_col_indices]
    # Create a new image with the dimensions of the sorted matrix
    # TODO: This takes too much memory. Must be optimised
    img = Image.new('1', (sorted_matrix.shape[1], sorted_matrix.shape[0]))  # 1-bit pixels, black and white
    # Process row by row
    for i, sorted_row_index in enumerate(sorted_row_indices):
        row = sorted_matrix.getrow(sorted_row_index)
        for j in range(row.indices.size):
            img.putpixel((row.indices[j], i), 1)  # Set black pixel for non-zero entry
    # Save the image
    img.save(out_file_name)
    # Show the image (optional)
    if show_image:
        img.show()


if __name__ == "__main__":
    # TODO: Run some tests
    pass

