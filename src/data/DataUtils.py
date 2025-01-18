import os

import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np


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


def load_or_create(raw_data_file, sparse_data_file, force_reprosess=False):
    # (Re)create the sparse matrix
    if force_reprosess or not os.path.isfile(sparse_data_file):
        sparse_matrix = load_data(raw_data_file)
    else:
        # TODO: implement this properly
        sparse_matrix = sparse_data_file.load()


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
