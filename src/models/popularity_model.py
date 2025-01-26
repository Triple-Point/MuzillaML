import logging

import torch

from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PopularityModel(RecommenderModel):
    def __init__(self, data):
        super().__init__(data)
        # Get the artist indices (second row of the indices tensor)
        artist_indices = self.data.indices()[1]

        # Count the occurrences of each artist index
        artist_counts = torch.bincount(artist_indices, minlength=self.data.size(1))

        # Sort the artist counts in descending order (optional)
        sorted_counts, sorted_indices = torch.sort(artist_counts, descending=True)

        # Sort items by count in descending order
        self.popularity_list = sorted_indices


    def recommend_items(self, user, topn=10):
        # Recommend the most popular items that the user hasn't seen yet.
        recommendations = []
        for best in self.popularity_list:
            if best not in user:
                recommendations.append(best)
            if len(recommendations) >= topn:
                break
        return recommendations
