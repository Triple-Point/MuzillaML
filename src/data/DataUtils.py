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


def remove_random_values(user_tensor: torch.sparse_coo_tensor, num_remove: int = 10, seed: int = None) -> tuple[torch.sparse_coo_tensor, list[int]]:
    """
    Remove random non-zero values from a sparse COO tensor.

    Args:
        user_tensor (torch.sparse_coo_tensor): Input sparse COO tensor.
        num_remove (int): Number of random values to remove. Default 10. Note: A maximum of 10% of the items will be removed
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


def get_sorted_artists(user_id: int, sparse_tensor: torch.sparse_coo_tensor) -> tuple[list[int], list[float]]:
    # Extract the indices and values from the sparse tensor for the given user ID
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


if __name__ == "__main__":
    # TODO: Run some tests
    pass
