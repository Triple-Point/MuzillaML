import logging

import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
logger.info(f"Using device: {device}")


def remove_random_values(user_tensor: torch.Tensor, num_remove: int = 10, seed: int = None):
    """
    Remove random non-zero values from a sparse COO tensor.

    Args:
        user_tensor (torch.sparse_coo_tensor): Input sparse COO tensor.
        num_remove (int): Number of random values to remove.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        torch.sparse_coo_tensor: The updated sparse tensor with values removed.
        list: A sorted list of artists (column indices) sorted by album frequency in descending order.
    """
    if not user_tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse COO tensor.")

    if seed is not None:
        torch.manual_seed(seed)

    # Extract indices and values
    indices = user_tensor.indices()
    values = user_tensor.values()

    # Determine how many values to remove
    actual_remove = min(num_remove, int(values.numel() * 0.1))

    # Select random indices to remove
    if actual_remove > 0:
        remove_indices = torch.multinomial(values, actual_remove, replacement=False)
    else:
        remove_indices = torch.tensor([], dtype=torch.int64, device=user_tensor.device)

    # Get the rows, columns, and values to be removed
    removed_items = [
        (indices[0, idx].item(), indices[1, idx].item(), values[idx].item())
        for idx in remove_indices
    ]

    # Create new indices and values without the removed items
    keep_mask = torch.ones_like(values, dtype=torch.bool)
    keep_mask[remove_indices] = False
    new_indices = indices[:, keep_mask]
    new_values = values[keep_mask]

    # Sort removed items by value in descending order
    removed_items = sorted(removed_items, key=lambda x: x[2], reverse=True)
    removed_artists = [item[1] for item in removed_items]

    # Create the updated sparse tensor
    updated_tensor = torch.sparse_coo_tensor(new_indices, new_values, user_tensor.size(), device=user_tensor.device)

    return updated_tensor.coalesce(), removed_artists


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
    norm_diagonal = torch.sparse_coo_tensor(diagonal_indices, diagonal_values,
                                            size=(row_norms.size(0), row_norms.size(0)))

    # Normalize rows of the sparse tensor
    return torch.sparse.mm(norm_diagonal, sparse_tensor)


if __name__ == "__main__":
    # TODO: Run some tests
    pass

