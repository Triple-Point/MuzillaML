import csv
import json
import os
import pickle
import logging

import numpy as np
import torch
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    def __init__(self, name_file: str):
        self.id_to_artist = load_csv_to_dict(name_file)

        # Create a dictionary to keep track of occurrences of each artist name
        artist_count = {}

        for key, value in self.id_to_artist.items():
            if value in artist_count:
                artist_count[value] += 1
                # Append ZWSP based on occurrence count
                self.id_to_artist[key] = value + '​' * artist_count[value]
            else:
                artist_count[value] = 0
        self.artist_to_id = {value: key for key, value in self.id_to_artist.items()}

        return

        # Dump to file for debug purposes
        with open("data/id_to_artist.json", "w") as outfile:
            json.dump(self.id_to_artist, outfile)
        with open("data/artist_to_id.json", "w") as outfile:
            json.dump(self.artist_to_id, outfile)

    def artists_to_ids(self, artists: list[str]) -> list[int]:
        return [
            self.artist_to_id[artist] if artist in self.artist_to_id else -1 for artist in artists
        ]

    def ids_to_artists(self, ids: list[int]) -> list[str]:
        return [
            self.id_to_artist[i] if i in self.id_to_artist else "<???>" for i in ids
        ]
