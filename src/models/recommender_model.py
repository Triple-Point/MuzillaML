import logging
import os
import pickle
from abc import ABC, abstractmethod

import torch
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TODO: Save the model, and ensure it can be loaded without the overhead of the data. Probably should be abstract.

class RecommenderModel(ABC):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        self.data = data
        self.user_id_to_index_map = user_id_to_index_map
        self.artist_id_to_index_map = artist_id_to_index_map
        self.artist_index_to_id_map = {index: id_ for id_, index in artist_id_to_index_map.items()}
        self.num_artists = len(self.artist_index_to_id_map)

    @classmethod
    def from_files(cls, raw_data_filename: str, sparse_data_filename: str, force_reprocess: bool = False):
        data, user_id_to_index_map, artist_id_to_index_map = cls._load_data(raw_data_filename, sparse_data_filename,
                                                                            force_reprocess)
        return cls(data, user_id_to_index_map, artist_id_to_index_map)

    @staticmethod
    def load_raw_data(data_file: str):
        """
        Loads user-artist interaction data from a CSV file and converts it into a sparse COO tensor.
        Args: data_file (str): Path to the CSV file containing the interaction data.
              The file must have the following columns:
                             - `user_id`: Unique identifier for each user.
                             - `artist_id`: Unique identifier for each artist.
                             - `album_count`: Interaction count (e.g., number of albums listened to).
        Returns:
            torch.Tensor: A PyTorch sparse COO tensor with:
                          - Indices representing `user_id` and `artist_id`.
                          - Values representing the `album_count`.
                          - Size derived from the maximum `user_id` and `artist_id` values.
            - The function uses categorical encoding to convert `user_id` and `artist_id` into integer indices.
            - The resulting tensor is moved to the current device (`cpu` or `cuda`) as determined by PyTorch.
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file {data_file} not found.")

        df = pd.read_csv(data_file, header=None, names=['user_id', 'artist_id', 'album_count'])

        user_ids = df['user_id'].unique().tolist()
        artist_ids = df['artist_id'].unique().tolist()

        # Create dictionaries to map original IDs to model's indices
        user_id_to_index_map = {original_id: index for index, original_id in enumerate(user_ids)}
        artist_id_to_index_map = {original_id: index for index, original_id in enumerate(artist_ids)}

        # Replace original IDs with new indices in the dataframe
        df['user_index'] = df['user_id'].map(user_id_to_index_map)
        df['artist_index'] = df['artist_id'].map(artist_id_to_index_map)

        # Extract indices and values
        indices = [df['user_index'].values, df['artist_index'].values]
        values = df['album_count'].values

        # Create sparse tensor
        data = torch.sparse_coo_tensor(indices, values).coalesce()

        return data, user_id_to_index_map, artist_id_to_index_map

    @staticmethod
    def _load_data(raw_data_filename: str, sparse_data_filename: str, force_reprocess: bool = False):
        """
        Initialize the recommender model with the given data file(s).

        Args:
            raw_data_filename (str): Path to the raw data file (e.g., a CSV file) containing user-artist interactions.
            sparse_data_filename (str): Path to the serialized sparse matrix file (e.g., a `.pkl` file).
                If not already present, this will be created
            force_reprocess (bool, optional): If `True`, reprocess the raw data file even if the serialized file exists.
                                              Defaults to `False`.
        """
        if force_reprocess or not os.path.isfile(sparse_data_filename):
            logger.info(f"Reprocessing data from {raw_data_filename}")
            data, user_id_to_index_map, artist_id_to_index_map = RecommenderModel.load_raw_data(raw_data_filename)
            with open(sparse_data_filename, "wb") as f:
                pickle.dump(
                    (data, user_id_to_index_map, artist_id_to_index_map), f)
        else:
            logger.info(f"Loading data from file {sparse_data_filename}")
            with (open(sparse_data_filename, "rb") as f):
                data, user_id_to_index_map, artist_id_to_index_map = pickle.load(f)
        return data, user_id_to_index_map, artist_id_to_index_map

    @abstractmethod
    def recommend_items(self, artist_ids: torch.sparse_coo_tensor, topn=10) -> list[int]:
        """
        Compute and return the index of a new artist for the given artist_ids.

        :param artist_ids: Index or identifier of the artist_ids
        :param topn: number of recommendations to make
        :return: Ordered list of most recommended items
        """
        pass

    def recommend_items_list(self, artist_ids: list[int], album_counts=None, topn: int = 10, excluded_artists=None) -> \
            list[int]:
        """
        List-based wrapper for recommend_items
        :param artist_ids:
        :param album_counts:
        :param excluded_artists:
        :param topn:
        :return:
        """
        if excluded_artists is None:
            excluded_artists = []
        # Map to the model IDs
        artists = [self.artist_id_to_index_map[i] for i in artist_ids]
        # User is one-d so use 0 for their index internally
        indices = [[0] * len(artists), artists]
        if album_counts:
            values = album_counts
        else:
            values = [1] * len(artist_ids)
        user_tensor = torch.sparse_coo_tensor(indices, values, size=(1, self.num_artists),
                                              dtype=torch.float64).coalesce()
        recommendations = self.recommend_items(user_tensor, topn + len(excluded_artists))
        recommended_ids = [self.artist_index_to_id_map[r] for r in recommendations]
        artist_list = [r for r in recommended_ids if r not in excluded_artists]
        return artist_list[:topn]
