def average_precision(relevant_items, retrieved_items):
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

if __name__ == "__main__":
    # Some test cases
    test_cases = [
        {
            "relevant_items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "retrieved_items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "expected": 1,
            "description": "Identical Lists"
        },
        {
            "relevant_items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "retrieved_items": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "expected": 0,
            "description": "Non-overlapping Lists"
        },
        {
            "relevant_items": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "retrieved_items": [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
            "expected": 1.0,
            "description": "100% Overlap in wrong order"
        },
        {
            "relevant_items": [0, 4, 2, 9, 8, 7, 1, 3, 5, 6],
            "retrieved_items": [6, 11, 2, 13, 8, 17, 1, 13, 5, 10],
            "expected": (1+2/3+3/5+4/7+5/9)/10,
            "description": "Partial Overlap"
        }
    ]

    for case in test_cases:
        result = average_precision(case["relevant_items"], case["retrieved_items"])
        assert abs(result - case["expected"]) < 1e-2, f"Failed {case['description']}: expected {case['expected']} but got {result}"

    print("All test cases passed!")
