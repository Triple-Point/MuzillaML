import logging
import os
import torch
import numpy as np

from src.data.DataUtils import get_sorted_artists
from src.data.TensorUtils import normalize_L1_sparse_tensor, dense_to_sparse, normalize_L2_sparse_tensor
from src.models.recommender_model import RecommenderModel

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class TFIDFModel(RecommenderModel):
    def __init__(self, data, user_id_to_index_map=None, artist_id_to_index_map=None):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)
        self.idf_vector = None
        self.tfidf_matrix = self.compute_tfidf()
        # Normalize it row-wise, ready for the cosine similarity
        self.tfidf_matrix = normalize_L2_sparse_tensor(self.tfidf_matrix)

    def save_model(self, file_path=None):
        """Saves TF-IDF and similarity matrices to disk."""
        if file_path is None:
            file_path = self.model_path

        model_data = {
            "tfidf_matrix": self.tfidf_matrix,
            "idf_vector": self.idf_vector
        }
        torch.save(model_data, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path=None):
        """Loads TF-IDF and similarity matrices from disk."""
        if not os.path.exists(file_path):
            print(f"Model file {file_path} not found.")
            return False

        model_data = torch.load(file_path, map_location=self.data.device)
        self.idf_vector = model_data["idf_vector"]
        self.tfidf_matrix = model_data["tfidf_matrix"]
        print(f"Model loaded from {file_path}")

        return True

    def compute_tfidf(self):
        """Computes the TF-IDF matrix from the user-artist interaction sparse tensor."""
        # Ensure we use the same device as self.data
        device = self.data.device
        indices = self.data.indices()
        values = self.data.values()
        shape = self.data.shape

        # Compute TF (normalize artist counts per user)
        row_sum = torch.sparse.sum(self.data, dim=1).to_dense()  # Total artist count per user
        tf_values = values / row_sum[indices[0]]  # Normalize per user

        # Compute IDF
        document_frequency = torch.bincount(self.data.indices()[1], minlength=self.data.shape[1])
        num_users = shape[0]
        self.idf_vector = torch.log((num_users + 1) / (document_frequency + 1)).to(device) + 1  # Smoothing

        # Scale TF values directly instead of using a sparse IDF matrix
        tfidf_values = tf_values * self.idf_vector[indices[1]]  # Apply IDF directly to existing values

        # Construct sparse TF-IDF matrix
        return torch.sparse_coo_tensor(indices, tfidf_values, shape, device=device).coalesce()

    def get_similar_users(self, user: torch.Tensor) -> tuple[list[int], list[float]]:
        device = self.data.device

        # Normalize and ensure input tensor is on the same device
        user_tf = normalize_L1_sparse_tensor(user.to(device))
        user_tfidf = user_tf * self.idf_vector
        # Normalize this, too for the cosine calculation
        tfidf_norm = normalize_L2_sparse_tensor(user_tfidf.coalesce())

        # Compute similarity scores (dot product with similarity matrix)
        cosine_sim = torch.sparse.mm(self.tfidf_matrix, tfidf_norm.t()).to_dense().cpu().numpy().flatten()

        # Get top indices and scores sorted by similarity (descending)
        top_indices = np.argsort(cosine_sim)[::-1]  # Indices of the top_n values
        top_scores = cosine_sim[top_indices]  # Corresponding similarity values

        # Get top indices and scores sorted by similarity (descending)
        return top_indices.tolist(), top_scores  # Indices of the top_n values

    def recommend_items(self, artist_ids: torch.sparse_coo_tensor, topn=10) -> list[int]:
        """
        Recommend top-N similar artists based on input artist IDs.

        :param artist_ids: Sparse tensor representing the user's artists.
        :param topn: Number of recommendations to return.
        :return: List of recommended artist indices.
        """
        similar_users, _ = self.get_similar_users(artist_ids)

        # Extract user's existing artists
        user_artists = set(artist_ids.indices()[1].tolist())

        recommendations = []
        for similar_user in similar_users:
            similar_artists, _ = get_sorted_artists(similar_user, self.data)
            # TODO: Selection could be better - just picking new artists in index order
            new_recommendations = [idx for idx in similar_artists if idx not in user_artists]
            recommendations.extend(new_recommendations)
            if len(recommendations) >= topn:
                return recommendations[:topn]
            else:
                user_artists |= set(new_recommendations)

        raise ValueError("Unable to generate recommendations with the given inputs.")


def calculate_tfidf(users):
    # Calculate term frequency (TF)
    tf = users / users.sum(axis=1, keepdims=True)

    # Calculate document frequency (DF)
    df = np.count_nonzero(users, axis=0)

    # Calculate inverse document frequency (IDF)
    idf = np.log((1 + users.shape[0]) / (1 + df)) + 1

    # Calculate TF-IDF
    tfidf = tf * idf
    return tfidf


def test(users):
    # Convert array to numpy array for easier calculations
    users = np.array(users)
    # Use the dense calculation
    tfidf = calculate_tfidf(users)

    # Create a sparse model
    data = dense_to_sparse(users)
    model = TFIDFModel(data)

    # Compare the model values to those calculated by the TFIDF above
    model_tfidf = model.tfidf_matrix.to_dense().cpu().detach().numpy()
    assert (model_tfidf - tfidf).all() < 1e-8

    # For each row in the users, we should have a 100% cosine match
    for i, user in enumerate(users):
        similar_users = model.get_similar_users(dense_to_sparse([user]))
        assert i == similar_users[0]


def generate_random_lists(users, artists, std_dev, zero_prob=0.0):
    # Generate normally distributed random values
    random_values = np.random.normal(0, std_dev, (users, artists))
    # Take the absolute value and convert to integers
    random_integers = np.abs(random_values).astype(int)
    # Apply the zero probability
    zero_mask = np.random.choice([0, 1], size=(users, artists), p=[zero_prob, 1 - zero_prob])
    random_integers *= zero_mask
    return random_integers.tolist()


if __name__ == "__main__":
    # A simple dummy document dataset
    test([
        [1, 1, 1, 1, 1, 1, 1, 1, 0],
        [34, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 2, 3, 4, 5, 6, 7, 8, 0]
    ])
    test(generate_random_lists(5, 10, 4.5))
    test(generate_random_lists(15, 100, 4.5))
    test(generate_random_lists(50, 1000, 4.5))
    test(generate_random_lists(500, 10000, 4.5, 0.8))
