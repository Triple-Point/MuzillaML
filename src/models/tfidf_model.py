import joblib
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

from src.models.recommender_model import RecommenderModel


class TFIDFModel(RecommenderModel):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_matrix = self.tfidf_transformer.fit_transform(sparse_matrix)
        # joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')
        joblib.dump(self.tfidf_matrix, "save_file")
        self.tfidf_matrix = joblib.load("save_file")

    def recommend_items(self, artist_ids: torch.sparse_coo_tensor, topn=10):
        cosine_similarities = linear_kernel(self.tfidf_matrix[artist_ids], self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-topn:-1]
        return self.data['user_id'].unique()[related_docs_indices]
