import os
import pickle

import pandas as pd
import torch
from scipy.sparse import csr_matrix


def load_data(data_file):
    data = pd.read_csv(data_file)
    data.columns = ['user_id', 'artist_id', 'album_count']
    # Pivot the dataset to get a matrix form
    # pivot_table = data.pivot(index='user_id', columns='artist_id', values='count').fillna(0)
    row = data['user_id'].astype('category').cat.codes
    col = data['artist_id'].astype('category').cat.codes
    sparse_matrix = csr_matrix((data['album_count'], (row, col)),
                               shape=(len(data['user_id'].unique()), len(data['artist_id'].unique())))
    return sparse_matrix


def load_or_create(raw_data_file, sparse_data_file, force_reprocess=False):
    # (Re)create the sparse matrix
    if force_reprocess or not os.path.isfile(sparse_data_file):
        sparse_matrix = load_data(raw_data_file)
        with open(sparse_data_file, "wb") as f:
            pickle.dump(sparse_matrix, f)
    else:
        # Load the sparse matrix using pickle
        with open("matrix.pkl", "rb") as f:
            sparse_matrix = pickle.load(f)
    return sparse_matrix



def split_user(input_matrix):
    # Initialize output matrices with the same shape as the input
    output_matrix_1 = csr_matrix(input_matrix.shape)
    output_matrix_2 = csr_matrix(input_matrix.shape)

    # Convert the input matrix to a COOrdinate format for easy iteration
    input_coo = input_matrix.tocoo()

    # Alternate assignment to the two matrices
    toggle = True
    for i, j, v in zip(input_coo.row, input_coo.col, input_coo.data):
        if v != 0:
            if toggle:
                output_matrix_1[i, j] = v
            else:
                output_matrix_2[i, j] = v
            toggle = not toggle

    # Convert back to CSR format if needed
    output_matrix_1 = output_matrix_1.tocsr()
    output_matrix_2 = output_matrix_2.tocsr()
    return output_matrix_1, output_matrix_2


# Convert csr_matrix to PyTorch sparse tensor
def csr_to_torch_sparse_tensor(csr_mat):
    coo = csr_mat.tocoo()  # Convert to COO format
    indices = torch.tensor([coo.row, coo.col], dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float32)
    shape = coo.shape
    return torch.sparse_coo_tensor(indices, values, size=shape)


# Normalize sparse tensor row-wise
def normalize_sparse_tensor(sparse_tensor):
    # Compute L2 norm of each row
    row_norms = torch.sqrt(torch.sparse.sum(sparse_tensor.pow(2), dim=1).to_dense())
    row_norms = torch.where(row_norms == 0, torch.tensor(1.0, device=row_norms.device), row_norms)  # Avoid div by zero

    # Create a diagonal sparse tensor for normalization
    row_norms_inv = 1.0 / row_norms
    row_indices = torch.arange(row_norms.size(0), device=row_norms.device)
    diagonal_indices = torch.stack([row_indices, row_indices])
    diagonal_values = row_norms_inv
    norm_diagonal = torch.sparse_coo_tensor(diagonal_indices, diagonal_values, size=(row_norms.size(0), row_norms.size(0)))

    # Normalize rows of the sparse tensor
    return torch.sparse.mm(norm_diagonal, sparse_tensor)
