import numpy as np
import torch
import logging

from src.data.TensorUtils import normalize_L2_sparse_tensor
from src.data.DataUtils import get_sorted_artists
from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineModel(RecommenderModel):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)
        self.device = getattr(data, "device", torch.device("cpu"))
        # Normalize the data in prep for the cosine similarity calculation.
        self.norm_data = normalize_L2_sparse_tensor(self.data).to(self.device)

    def get_similar_users(self, user: torch.Tensor) -> tuple[list[int], list[float]]:
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
        norm_user = normalize_L2_sparse_tensor(user).to(self.device)

        cosine_sim = torch.sparse.mm(self.norm_data, norm_user.t())
        cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()

        # Get top indices and scores sorted by similarity (descending)
        top_indices = np.argsort(cosine_sim)[::-1]  # Indices of the top_n values
        top_scores = cosine_sim[top_indices]  # Corresponding similarity values

        return top_indices.tolist(), top_scores.tolist()

    def recommend_items(self, artist_ids: torch.Tensor, topn: int = 10) -> list[int]:
        """
        Recommend an artist for the user based on the most similar user's preferences.
        Args:
            :param artist_ids: User to get recommendation for
            :param topn: number of recommendations to make
        Returns:
            int: Sorted list of topn recommended artists.
        """
        similar_users, _ = self.get_similar_users(artist_ids)

        # Extract user's existing artists
        user_artists = set(artist_ids.indices()[1].tolist())

        recommendations = []
        for similar_user in similar_users:
            similar_artists, _ = get_sorted_artists(similar_user, self.data)
            new_recommendations = [idx for idx in similar_artists if idx not in user_artists]
            recommendations.extend(new_recommendations)
            if len(recommendations) >= topn:
                return recommendations[:topn]
            else:
                user_artists |= set(new_recommendations)

        raise ValueError("Unable to generate recommendations with the given inputs.")
