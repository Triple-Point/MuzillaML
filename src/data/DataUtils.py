import logging
import os
import pickle
import random

import pandas as pd
import numpy as np
from PIL import Image
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_file: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    # Load the data
    logger.info(f"(Re)Loading and processing the data")
    data = pd.read_csv(data_file)
    data.columns = ['user_id', 'artist_id', 'album_count']

    # Convert user_id and artist_id to categorical codes
    row = torch.tensor(data['user_id'].astype('category').cat.codes, dtype=torch.int64, device=device)
    col = torch.tensor(data['artist_id'].astype('category').cat.codes, dtype=torch.int64, device=device)
    values = torch.tensor(data['album_count'].values, dtype=torch.float32, device=device)

    # Determine the shape of the sparse tensor
    num_rows = row.max().item() + 1
    num_cols = col.max().item() + 1

    logger.info(f"Making COO tensor")
    # Create COO sparse tensor
    coo_tensor = torch.sparse_coo_tensor(
        indices=torch.stack([row, col]),
        values=values,
        size=(num_rows, num_cols),
        device=device
    )

    logger.info(f"Calling COO coalesce")
    return coo_tensor.coalesce()


def load_or_create(raw_data_file, sparse_data_file, force_reprocess=False):
    # (Re)create the sparse matrix
    if force_reprocess or not os.path.isfile(sparse_data_file):
        sparse_matrix = load_data(raw_data_file)
        with open(sparse_data_file, "wb") as f:
            pickle.dump(sparse_matrix, f)
    else:
        # Load the sparse matrix using pickle
        with open(sparse_data_file, "rb") as f:
            sparse_matrix = pickle.load(f)
    return sparse_matrix


# TODO: Add a seed variable so tests can be reproduced. (Also write tests)
def remove_random_values(user_tensor, num_remove=10):
    """
    Remove random non-zero values from a torch.tensor.

    Args:
        user_tensor (torch.tensor): Input dense tensor.
        num_remove (int): Number of random values to remove.

    Returns:
        torch.tensor: The updated dense tensor with values removed.
        list: A sorted list of removed items in the form [(row, col, value), ...],
              sorted by value in descending order.
    """
    if user_tensor.is_sparse:
        raise ValueError("Input tensor must be dense.")

    # Find all non-zero entries
    non_zero_indices = torch.nonzero(user_tensor, as_tuple=False)
    non_zero_values = user_tensor[non_zero_indices[:, 0], non_zero_indices[:, 1]]

    # Determine how many values to remove
    actual_remove = min(num_remove, int(non_zero_values.numel() * 0.1))

    # Select random indices to remove
    if actual_remove > 0:
        remove_indices = random.sample(range(non_zero_values.numel()), actual_remove)
    else:
        remove_indices = []

    # Get the rows, columns, and values to be removed
    removed_items = []
    for idx in remove_indices:
        row, col = non_zero_indices[idx].tolist()
        value = user_tensor[row, col].item()
        removed_items.append((row, col, value))
        user_tensor[row, col] = 0  # Remove the value

    # Sort removed items by value in descending order
    removed_items = sorted(removed_items, key=lambda x: x[2], reverse=True)
    removed_artists = [x[1] for x in removed_items]

    return user_tensor, removed_artists


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
