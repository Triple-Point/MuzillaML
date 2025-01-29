from abc import ABC, abstractmethod
from typing import List

import torch


class RecommenderModel(ABC):
    def __init__(self, data: torch.sparse_coo_tensor):
        """
        Initialize the recommender model with the given data.

        :param data: Data used for recommendations, e.g., user-item matrix
        """
        self.data = data

    @abstractmethod
    def recommend_items(self, user: torch.sparse_coo_tensor, topn=10) -> List[int]:
        """
        Compute and return the index of a new artist for the given user.

        :param user: Index or identifier of the user
        :param topn: number of recommendations to make
        :return: Ordered list of most recommended items
        """
        pass

    def recommend_items_list(self, user: List[int], album_counts=None, topn: int = 10) -> List[int]:
        """
        List-based wrapper for recommend_items
        :param album_counts:
        :param user:
        :param topn:
        :return:
        """
        indices = [[0] * len(user), user]
        if album_counts:
            values = album_counts
        else:
            values = [1] * len(user)
        user_tensor = torch.sparse_coo_tensor(indices, values, size=(1, 522106), dtype=torch.float64).coalesce()
        return self.recommend_items(user_tensor, topn)
