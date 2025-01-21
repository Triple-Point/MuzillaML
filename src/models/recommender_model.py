from abc import ABC, abstractmethod

class RecommenderModel(ABC):
    def __init__(self, data):
        """
        Initialize the recommender model with the given data.

        :param data: Data used for recommendations, e.g., user-item matrix
        """
        self.data = data

    @abstractmethod
    def get_similar_user_index(self, user):
        """
        Compute and return the index of the most similar user to the given user.

        :param user: Index or identifier of the user
        :return: Index of the most similar user
        """
        pass

    @abstractmethod
    def recommend_artist(self, user):
        """
        Compute and return the index of a new artist for the given user.

        :param user: Index or identifier of the user
        :return: Index of the most similar user
        """
        pass

    @abstractmethod
    def evaluate(self, user):
        """
        Compute and return the quality of the model on the test set.

        :param user: Index or identifier of the user
        :return: Evaluation score
        """
        pass
