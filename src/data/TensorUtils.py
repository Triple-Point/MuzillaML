import logging

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def distribute_sparse_tensor(sparse_tensor: torch.Tensor, num_buckets: int = 10) -> list[torch.Tensor]:
    """
    Distribute the users from a collated torch.sparse_coo_tensor into `num_buckets` tensors cyclically.

    Args:
        sparse_tensor (torch.sparse_coo_tensor): Input sparse tensor in COO format.
        num_buckets (int): Number of output tensors (buckets).

    Returns:
        list: A list of torch.sparse_coo_tensor, one for each bucket.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse COO tensor.")
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer.")

    sub_tensors = []

    for mod_id in range(num_buckets):
        logger.info(f"Masking for tensor {mod_id}")

        # Get mask for indices[0] % num_sub_tensors == mod_id
        mask = (sparse_tensor.indices()[0] % num_buckets) == mod_id

        # Slice indices and values
        sub_indices = sparse_tensor.indices()[:, mask]
        sub_values = sparse_tensor.values()[mask]

        # Create sub-tensor
        sub_tensors.append(torch.sparse_coo_tensor(
            sub_indices, sub_values, sparse_tensor.size(), device=device
        ).coalesce())
        logger.info(f"Tensor {mod_id} contains {sub_tensors[-1].indices().shape[1]} non-zero values")
    return sub_tensors


def concatenate_except_one(sub_tensors, excluded_index):
    """
    Concatenate all tensors except the one at the excluded index.

    Args:
        sub_tensors (list of torch.sparse_coo_tensor): List of sparse tensors.
        excluded_index (int): Index of the tensor to exclude.

    Returns:
        torch.sparse_coo_tensor: The concatenated sparse tensor.
    """
    indices_list = []
    values_list = []
    size = sub_tensors[0].size()
    # Ensure all tensors have the same size
    if not all(tensor.size(1) == size[1] for tensor in sub_tensors):
        raise ValueError("All tensors must have the same number of columns.")

    total_rows = 0

    for i, tensor in enumerate(sub_tensors):
        if i == excluded_index:
            continue

        # Offset row indices by the current total_rows
        indices = tensor.indices()
        indices[0] += total_rows
        indices_list.append(indices)

        values_list.append(tensor.values())

        # Update total rows
        total_rows += tensor.size(0)

    # Concatenate indices and values
    if indices_list:
        concatenated_indices = torch.cat(indices_list, dim=1)
        concatenated_values = torch.cat(values_list)
    else:
        concatenated_indices = torch.empty((2, 0), dtype=torch.int64)
        concatenated_values = torch.empty(0, dtype=torch.float32)

    # Create the concatenated sparse tensor
    concatenated_size = (total_rows, size[1])
    concatenated_tensor = torch.sparse_coo_tensor(concatenated_indices, concatenated_values, size=concatenated_size)

    return concatenated_tensor.coalesce()


import torch


def get_sorted_artists(user_id, sparse_tensor):
    # Extract the indices and values from the sparse tensor
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    # Locate the positions in the indices corresponding to the given user_id
    target_user_indices = indices[0] == user_id
    user_artist_indices = indices[1][target_user_indices]

    # Extract the corresponding values for the user
    user_artist_values = values[indices[0] == user_id]

    # Sort indices based on the values in descending order
    sorted_indices = torch.argsort(user_artist_values, descending=True)

    # Retrieve the sorted artists and their corresponding values
    sorted_artists = user_artist_indices[sorted_indices]
    sorted_values = user_artist_values[sorted_indices]

    return sorted_artists, sorted_values


def get_users(sparse_tensor):
    """
    Extracts per-user sparse tensors from a given sparse COO tensor.

    Args:
        sparse_tensor (torch.sparse_coo_tensor): Input sparse COO tensor.

    Yields:
        torch.sparse_coo_tensor: A sparse tensor for each user, containing only their non-zero interactions.

    Raises:
        ValueError: If the input tensor is not in sparse COO format.
    """
    if not sparse_tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse COO tensor.")

    # Iterate over each unique user ID
    user_ids = torch.unique(sparse_tensor.indices()[0])

    for user_id in user_ids:
        # Get mask for current user ID
        mask = sparse_tensor.indices()[0] == user_id

        # Slice the indices and values for the current user ID
        user_indices = sparse_tensor.indices()[:, mask]
        user_values = sparse_tensor.values()[mask]

        # Adjust the row index to start from 0 for the user's tensor
        user_indices[0] = user_indices[0] - user_id  # Normalize user row index to 0

        # Create a sparse tensor for the user
        user_sparse = torch.sparse_coo_tensor(
            indices=user_indices,
            values=user_values,
            size=(1, sparse_tensor.size(1)),  # Single row for the user
            device=device
        ).coalesce()

        yield user_sparse
