import numpy as np
import torch

from src.models.recommender_model import RecommenderModel


class CosineModel(RecommenderModel):
    def __init__(self, sparse_data):
        super().__init__(sparse_data)

    # Compute cosine similarity using sparse tensors
    def sparse_cosine_similarity(self, norm_tensor1):
        # Compute cosine similarity (sparse matrix multiplication)
        return torch.sparse.mm(norm_tensor1, self.data.t())

    def get_similar_user_index(self, new_user):
        # Compute cosine similarity
        cosine_sim = self.sparse_cosine_similarity(new_user)

        # Convert result back to dense for inspection if needed
        cosine_sim = cosine_sim.to_dense().cpu().numpy().flatten()
        #print(cosine_sim.cpu().numpy())

        # Find the index of the most similar user
        most_similar_user_index = np.argmax(cosine_sim)
        similarity_value = cosine_sim[most_similar_user_index]
        #print(f"Cosine Similarities {cosine_sim.shape}:\n{cosine_sim}")
        print(f"{most_similar_user_index=}\t{similarity_value=}")
        return most_similar_user_index, similarity_value
