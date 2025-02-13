from scipy.sparse import csr_matrix

from src.data.TensorUtils import dense_to_sparse
from src.models.cosine_model import CosineModel

# 2 users with 6 artists
test_data = dense_to_sparse([[0, 1, 2, 0, 3, 4],
                             [4, 0, 3, 2, 1, 0]])
model = CosineModel(test_data)

# Same user#0, but missing artist #2
user = [0, 1, 0, 0, 3, 4]
new_artist = model.recommend_items(dense_to_sparse([user]), topn=1)
print(new_artist)

# Similar to user#1, but missing artist #3
user = [4, 0, 3, 0, 1, 1]
new_artist = model.recommend_items(dense_to_sparse([user]), topn=1)
print(new_artist)
