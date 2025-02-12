import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def concatenate_except_one(sub_tensors: list[torch.sparse_coo_tensor], excluded_index: int) -> torch.sparse_coo_tensor:
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
    if not all(tensor.is_sparse for tensor in sub_tensors):
        raise ValueError("Input tensors must all be a sparse COO tensor")

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


def get_all_users(sparse_tensor: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
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


# Normalize sparse tensor row-wise
def normalize_L1_sparse_tensor(sparse_tensor: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    """
    Performs row-wise L1 normalization (each row sums to 1) for a sparse matrix.

    :param sparse_tensor: Input torch.sparse_coo_tensor of shape (num_rows, num_cols).
    :return: Row-normalized sparse tensor where each row sums to 1.
    """
    device = sparse_tensor.device
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()

    # Compute row sum (L1 norm for each row)
    row_sums = torch.sparse.sum(sparse_tensor, dim=1).to_dense()  # Shape: (num_rows,)

    # Prevent division by zero (for rows with all zeros)
    row_sums = row_sums.clamp(min=1e-8)  # Avoid NaN

    # Normalize values (divide each element by its corresponding row sum)
    normalized_values = values / row_sums[indices[0]]  # Use row indices to index row_sums

    # Construct new sparse tensor
    return torch.sparse_coo_tensor(indices, normalized_values, sparse_tensor.shape, device=device)


def normalize_L2_sparse_tensor(sparse_tensor: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    if not sparse_tensor.is_sparse:
        raise ValueError(f"Input tensor must be a sparse COO tensor, not {sparse_tensor}")

    # Compute L2 norm of each row
    row_norms = torch.sqrt(torch.sparse.sum(sparse_tensor.pow(2), dim=1).to_dense())
    row_norms = torch.where(row_norms == 0, torch.tensor(1.0, device=row_norms.device), row_norms)  # Avoid div by zero

    # Create a diagonal sparse tensor for normalization
    row_norms_inv = 1.0 / row_norms
    row_indices = torch.arange(row_norms.size(0), device=row_norms.device)
    diagonal_indices = torch.stack([row_indices, row_indices])
    diagonal_values = row_norms_inv
    norm_diagonal = torch.sparse_coo_tensor(diagonal_indices, diagonal_values,
                                            size=(row_norms.size(0), row_norms.size(0)))

    # Normalize rows of the sparse tensor
    return torch.sparse.mm(norm_diagonal, sparse_tensor)


def dense_to_sparse(dense_matrix):
    # Explicitly construct the indices and values for demonstration purposes
    # Expects a 2D input
    indices = [[], []]
    values = []
    num_users = len(dense_matrix)
    num_artists = 0
    for row, user_data in enumerate(dense_matrix):
        for column, artist_count in enumerate(user_data):
            if artist_count > 0:
                indices[0].append(row)
                indices[1].append(column)
                values.append(artist_count)
            if num_artists <= column:
                num_artists = column + 1
    return torch.sparse_coo_tensor(indices, values, (num_users, num_artists)).coalesce()


if __name__ == "__main__":
    un_norm = dense_to_sparse([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3]])
    norm = normalize_L1_sparse_tensor(un_norm).to_dense()
    logger.info(norm)
