import logging
from random import randrange
from src.models.recommender_model import RecommenderModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RandomModel(RecommenderModel):
     def recommend_items(self, artist_ids, topn=10):
        # Recommend the most popular items that the user hasn't seen yet at random
        # Return a list of size topn chosen at random from the range
        return [randrange(self.num_artists) for _ in range(topn)]
