def mean_average_precision(relevant_items, retrieved_items):
    """
    Calculate the Mean Average Precision (MAP).

    Args:
        relevant_items (list): A list of relevant items (ground truth).
        retrieved_items (list): A list of retrieved items (ordered by relevance).

    Returns:
        float: The MAP score.
    """
    if not relevant_items or not retrieved_items:
        return 0.0

    average_precisions = []
    relevant_found = 0

    for idx, item in enumerate(retrieved_items):
        if item in relevant_items:
            relevant_found += 1
            precision_at_k = relevant_found / (idx + 1)
            average_precisions.append(precision_at_k)

    # MAP is the mean of the average precisions for all relevant items
    return sum(average_precisions) / len(relevant_items) if relevant_items else 0.0
