import torch

# Compute cosine similarity using sparse tensors
def sparse_cosine_similarity(norm_tensor1, norm_tensor2):
    # Compute cosine similarity (sparse matrix multiplication)
    return torch.sparse.mm(norm_tensor1, norm_tensor2.t())
