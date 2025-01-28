from typing import List, Tuple
import numpy as np
import torch
import logging

from src.data.DataUtils import normalize_sparse_tensor
from src.data.TensorUtils import get_sorted_artists
from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineModel(RecommenderModel):
    def __init__(self, data):
        super().__init__(data)
        self.device = getattr(data, "device", torch.device("cpu"))
        # Normalize the data in prep for the cosign similarity calculation. This could take a while...
        self.norm_data = normalize_sparse_tensor(self.data).to(self.device)

    def sparse_cosine_similarity(self, norm_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity using sparse tensors.
        Args:
            norm_tensor (Tensor): Normalized user tensor.
        Returns:
            Tensor: Cosine similarity tensor.
        """
        return torch.sparse.mm(self.norm_data, norm_tensor.t())

    def get_similar_users(self, user: torch.Tensor) -> Tuple[List[int], List[float]]:
        """
        Compute the indices and similarity values of the most similar users based on cosine similarity.

        Args:
            user (torch.Tensor): User tensor.

        Returns:
            Tuple[List[int], List[float]]:
                - List of indices of the top similar users, sorted in descending order of similarity.
                - List of corresponding similarity values, also sorted in descending order.
        """
        # Compute cosine similarity
        cosine_sim = self.sparse_cosine_similarity(user)
        cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()

        # Get top indices and scores sorted by similarity (descending)
        top_indices = np.argsort(cosine_sim)[::-1]  # Indices of the top_n values
        top_scores = cosine_sim[top_indices]  # Corresponding similarity values

        return top_indices.tolist(), top_scores.tolist()

    def recommend_items(self, user: torch.Tensor, topn: int = 10) -> List[int]:
        """
        Recommend an artist for the user based on the most similar user's preferences.
        Args:
            :param user: User to get recommendation for
            :param topn: number of recommendations to make
        Returns:
            int: Sorted list of topn recommended artists.
        """
        norm_user = normalize_sparse_tensor(user).to(self.device)
        # Extract user's existing artists
        user_artists = set(user.indices()[1].tolist())

        similar_users, _ = self.get_similar_users(norm_user)

        recommendations = []
        for similar_user in similar_users:
            similar_artists, _ = get_sorted_artists(similar_user, self.data)
            new_recommendations = [idx for idx in similar_artists.tolist() if idx not in user_artists]
            recommendations.extend(new_recommendations)
            if len(recommendations) >= topn:
                return recommendations[:topn]
            else:
                user_artists |= set(new_recommendations)

        raise ValueError("Unable to generate recommendations with the given inputs.")

    def recommend_items_list(self, user: List[int], topn: int = 10):
        return self.recommend_items(user, topn)