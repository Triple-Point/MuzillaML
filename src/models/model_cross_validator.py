# Wrapper for a recommender model that will perform cross validation on the given dataset
import logging

import torch

from src.data.DataUtils import remove_random_values
from src.data.TensorUtils import get_all_users, concatenate_except_one
from src.metrics.AveragePrecision import average_precision
from src.models.model_factory import model_factory
from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda"


def evaluate_model(model, test_tensor):
    """
    Evaluate the model using mean average precision.

    Args:
        model: The trained model to evaluate.
        test_tensor: Sparse tensor for test data.

    Returns:
        float: Mean average precision score for the evaluation.
    """
    total_score = 0
    i = 0
    for i, user in enumerate(get_all_users(test_tensor)):
        # TODO: Batch some of this to save time?
        masked_user, masked_artists = remove_random_values(user)
        # Generate the top-n artists
        recommended_artists = model.recommend_items(masked_user, len(masked_artists))
        total_score += average_precision(recommended_artists, masked_artists)
        average_score = total_score / (i + 1)
        logger.info(f'{i=}\t{total_score=}\t{average_score=}')
    logger.info(f"Tested users {i} in the set of size {test_tensor.size()}")
    # Mean Average Precision
    return total_score / (i + 1)


class CrossValidator(RecommenderModel):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)

    def set_cross_validation_params(self, model_type, num_folds):
        self.model_type = model_type
        self.num_folds = num_folds
        logger.info(f"Splitting into {self.num_folds=} cross-validation sets")
        self.buckets = []
        self.create_buckets()

    def recommend_items(self, artist_ids: torch.sparse_coo_tensor, topn=10) -> list[int]:
        # Dummy implementation
        return []

    def evaluate(self):

        model_class = model_factory(self.model_type)

        eval_scores = []
        for i, test_tensor in enumerate(self.buckets):
            logger.info(f"Starting a cross-validation batch {i}")
            train_tensor = concatenate_except_one(self.buckets, i)
            # Make a model on the train data
            logger.debug(f"Creating model")
            # Inject the data directly into the model
            model = model_class(train_tensor, self.user_id_to_index_map, self.artist_id_to_index_map)
            logger.debug(f"About to evaluate")
            score = evaluate_model(model, test_tensor)
            eval_scores.append(score)
        return eval_scores

    def create_buckets(self):
        """
        Distribute the users from a collated torch.sparse_coo_tensor into `self.num_folds` tensors cyclically.
        """
        if self.num_folds <= 0:
            raise ValueError(f"{self.num_folds=} but must be a positive integer. Got: ")

        self.buckets = []

        for mod_id in range(self.num_folds):
            logger.info(f"Masking for tensor {mod_id}")

            # Get mask for indices[0] % num_sub_tensors == mod_id
            mask = (self.data.indices()[0] % self.num_folds) == mod_id

            # Slice indices and values
            sub_indices = self.data.indices()[:, mask]
            sub_values = self.data.values()[mask]

            # Create sub-tensor
            self.buckets.append(torch.sparse_coo_tensor(
                sub_indices, sub_values, self.data.size(), device=device
            ).coalesce())
            logger.info(f"Tensor {mod_id} contains {self.buckets[-1].indices().shape[1]} non-zero values")

