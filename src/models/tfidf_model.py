import joblib
import torch
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel

from src.models.recommender_model import RecommenderModel


class TFIDFModel(RecommenderModel):
    def __init__(self, sparse_matrix, save_file='tfidf_matrix.pkl'):
        super().__init__(sparse_matrix)
        self.data = sparse_matrix
        self.tfidf_transformer = TfidfTransformer()
        self.tfidf_matrix = self.tfidf_transformer.fit_transform(sparse_matrix)
        # joblib.dump(tfidf_transformer, 'tfidf_transformer.pkl')
        joblib.dump(self.tfidf_matrix, save_file)
        self.tfidf_matrix = joblib.load(save_file)

    def recommend_items(self, user: torch.sparse_coo_tensor, topn=10):
        cosine_similarities = linear_kernel(self.tfidf_matrix[user], self.tfidf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-topn:-1]
        return self.data['user_id'].unique()[related_docs_indices]
