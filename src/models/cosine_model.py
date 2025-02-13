import numpy as np
import torch
import logging

from src.data.TensorUtils import normalize_L2_sparse_tensor
from src.models.recommender_model import SimilarUserRecommender

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CosineModel(SimilarUserRecommender):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)
        self.device = getattr(data, "device", torch.device("cpu"))
        # Normalize the data in prep for the cosine similarity calculation.
        self.norm_data = normalize_L2_sparse_tensor(self.data).to(self.device)

    def get_similar_users(self, user: torch.Tensor) -> tuple[list[int], list[float]]:
        # Compute cosine similarity
        norm_user = normalize_L2_sparse_tensor(user).to(self.device)

        cosine_sim = torch.sparse.mm(self.norm_data, norm_user.t())
        cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()

        # Get top indices and scores sorted by similarity (descending)
        top_indices = np.argsort(cosine_sim)[::-1]  # Indices of the top_n values
        top_scores = cosine_sim[top_indices]  # Corresponding similarity values

        return top_indices.tolist(), top_scores.tolist()

