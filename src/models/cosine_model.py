from typing import Tuple
import numpy as np
import torch
import logging

from torch import Tensor

from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineModel(RecommenderModel):
    def __init__(self, data):
        super().__init__(data)

    def sparse_cosine_similarity(self, norm_tensor1: Tensor) -> Tensor:
        """
        Compute cosine similarity using sparse tensors.
        Args:
            norm_tensor1 (Tensor): Normalized user tensor.
        Returns:
            Tensor: Cosine similarity tensor.
        """
        return torch.sparse.mm(norm_tensor1, self.data.t())

    def get_similar_user_index(self, user: Tensor) -> Tuple[int, float]:
        """
        Compute the index of the most similar user based on cosine similarity.
        Args:
            user (Tensor): User tensor.
        Returns:
            Tuple[int, float]: Index of the most similar user and similarity value.
        """
        cosine_sim = self.sparse_cosine_similarity(user)
        cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()
        logger.info(cosine_sim)

        most_similar_user_index = np.argmax(cosine_sim)
        similarity_value = cosine_sim[most_similar_user_index]
        logger.debug(f"Cosine Similarities {cosine_sim.shape}:\n{cosine_sim}")
        logger.debug(f"{most_similar_user_index=}\t{similarity_value=}")
        return most_similar_user_index, similarity_value

    def recommend_items(self, user: Tensor) -> int:
        """
        Recommend an artist for the user based on the most similar user's preferences.
        Args:
            user (Tensor): User tensor.
            :param artist_count: number of recommendations to make
        Returns:
            int: Index of the recommended artist.
        """
        # TODO: Pre-normalize these in the cross_validate function
        #  Convert to PyTorch sparse tensors and move to GPU
        #train_tensor = normalize_sparse_tensor(self.data).to('cuda')
        #test_tensor = normalize_sparse_tensor(test_tensor).to('cuda')
        #masked_user = normalize_sparse_tensor(user.to('cuda'))

        similar_user_index, _ = self.get_similar_user_index(user)
        similar_user = self.data[similar_user_index].flatten()

        indices_vector1 = set(user[0]._indices()[0].tolist())
        indices_vector2 = set(similar_user._indices()[0].tolist())

        unique_indices_vector2 = indices_vector2 - indices_vector1

        largest_value = 0
        artist_idx = None
        for idx in unique_indices_vector2:
            value = similar_user._values()[similar_user._indices()[0] == idx].item()
            if value > largest_value:
                largest_value = value
                artist_idx = idx

        logger.info(f"Largest remaining value in vector 2: {largest_value}")
        return artist_idx
