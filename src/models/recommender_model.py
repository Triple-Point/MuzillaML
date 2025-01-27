from abc import ABC, abstractmethod
import torch


class RecommenderModel(ABC):
    def __init__(self, data: torch.sparse_coo_tensor):
        """
        Initialize the recommender model with the given data.

        :param data: Data used for recommendations, e.g., user-item matrix
        """
        self.data = data

    @abstractmethod
    def recommend_items(self, user: torch.sparse_coo_tensor, topn=10):
        """
        Compute and return the index of a new artist for the given user.

        :param user: Index or identifier of the user
        :param topn: number of recommendations to make
        :return: Ordered list of most recommended items
        """
        pass
