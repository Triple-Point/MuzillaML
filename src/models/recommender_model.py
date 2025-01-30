from abc import ABC, abstractmethod
from typing import List

import torch


class RecommenderModel(ABC):
    def __init__(self, data: torch.sparse_coo_tensor, artist_id_to_index_map=None, user_lookup=None):
        """
        Initialize the recommender model with the given data.

        :param data: Data used for recommendations, e.g., artist_ids-item matrix
        :param data: artist_id_to_index_map (optional) used for the functions recommend_items_list
        :param data: user_lookup (optional) Currently unused
        """
        self.data = data
        self.artist_id_to_index_map = artist_id_to_index_map
        if artist_id_to_index_map is not None:
            self.artist_index_to_id_map = {index: id_ for id_, index in self.artist_id_to_index_map.items()}
        else:
            self.artist_index_to_id_map = None
        self.user_lookup = user_lookup


    @abstractmethod
    def recommend_items(self, user: torch.sparse_coo_tensor, topn=10) -> List[int]:
        """
        Compute and return the index of a new artist for the given artist_ids.

        :param user: Index or identifier of the artist_ids
        :param topn: number of recommendations to make
        :return: Ordered list of most recommended items
        """
        pass

    def recommend_items_list(self, artist_ids: List[int], album_counts=None, topn: int = 10) -> List[int]:
        """
        List-based wrapper for recommend_items
        :param album_counts:
        :param artist_ids:
        :param topn:
        :return:
        """
        # Map to the model IDs
        artists = [self.artist_id_to_index_map[i] for i in artist_ids]
        # User is one-d so use 0 for their index internally
        indices = [[0] * len(artists), artists]
        if album_counts:
            values = album_counts
        else:
            values = [1] * len(artist_ids)
        # TODO Nagic Mumber!!
        user_tensor = torch.sparse_coo_tensor(indices, values, size=(1, 522106), dtype=torch.float64).coalesce()
        recommendations = self.recommend_items(user_tensor, topn)
        artist_list = [self.artist_index_to_id_map[r] for r in recommendations]
        return artist_list
