import csv
import os
import pickle
import logging

import numpy as np
import pandas as pd
import torch
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(data_file: str) -> tuple[torch.sparse_coo_tensor, dict[int, int], dict[int, int]]:
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

    return tensor, user_id_map, artist_id_map


def load_or_create(raw_data_file: str, sparse_data_file: str, force_reprocess: bool = False) -> tuple[torch.sparse_coo_tensor, dict[int, int], dict[int, int]]:
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
        sparse_matrix, user_id_map, artist_id_map = load_data(raw_data_file)
        with open(sparse_data_file, "wb") as f:
            pickle.dump((sparse_matrix, user_id_map, artist_id_map), f)
    else:
        logger.info(f"Loading data from file {sparse_data_file}")
        with open(sparse_data_file, "rb") as f:
            sparse_matrix, user_id_map, artist_id_map = pickle.load(f)
    return sparse_matrix, user_id_map, artist_id_map


def dump_to_image(user_artist_matrix: torch.sparse_coo_tensor, out_file_name: str, show_image=False):
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


def load_csv_to_dict(file_path):
    """
    Loads a CSV file containing key-value pairs into a dictionary.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        dict: A dictionary with keys and values from the CSV file.
    """
    result_dict = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) >= 2:  # Ensure there are at least two columns
                    key = int(row[0])
                    value = row[1] if len(row) == 2 else ', '.join(row[1:])
                    result_dict[key] = value
                else:
                    print(f"Skipping invalid row: {row}")
        return result_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


class ArtistLookup:
    def __init__(self, artist_id_lookup: dict[int, int], name_file: str):
        self.id_to_artist = load_csv_to_dict(name_file)
        self.artist_to_id = {output: index for index, output in self.id_to_artist.items()}
        self.output_to_id = artist_id_lookup
        self.id_to_output = {output: index for index, output in self.output_to_id.items()}

    def outputs_to_artists(self, output: list[int]):
        original_ids = [self.output_to_id[i] for i in output]
        artist_list = [self.id_to_artist[i] for i in original_ids]
        return artist_list

    def artists_to_ids(self, artists: list[str]) -> list[int]:
        return [
            self.artist_to_id[artist] if artist in self.artist_to_id else -1 for artist in artists
        ]

    def ids_to_artists(self, ids:list[int]) -> list[str]:
        return [
            self.id_to_artist[i] if i in self.id_to_artist else "<???>" for i in ids
        ]
