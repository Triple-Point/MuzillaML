import logging
import torch
from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class PopularityModel(RecommenderModel):
    def __init__(self, data, user_id_to_index_map, artist_id_to_index_map):
        super().__init__(data, user_id_to_index_map, artist_id_to_index_map)
        # Get the artist indices (second row of the indices' tensor)
        artist_ids = self.data.indices()[1]

        # Count the occurrences of each artist index
        artist_counts = torch.bincount(artist_ids, minlength=self.data.size(1))

        # Sort the artist counts in descending order (optional)
        sorted_counts, sorted_ids = torch.sort(artist_counts, descending=True)

        # Sort items by count in descending order
        self.popularity_list = sorted_ids.tolist()

    def recommend_items(self, artist_ids, topn=10):
        # Recommend the most popular items that the user hasn't seen yet.
        user_artists = artist_ids.indices()[1].tolist()
        recommendations = []
        for best in self.popularity_list:
            if best not in user_artists:
                recommendations.append(best)
            if len(recommendations) >= topn:
                break
        return recommendations
