RecommenderModel is the interface.

The Muzilla data is already essentially in a Sparse COO matrix, so this is the expected format of the input data. The Output is a list of integers, ordered with the best recommendation first. 

Children need to implement the single abstract function:

        def recommend_items(self, artist_ids: torch.sparse_coo_tensor, topn=10) -> list[int]:
        """
        Compute and return the index of a new artist for the given artist_ids.

        :param artist_ids: Index of the artist_ids
        :param topn: number of recommendations to make
        :return: Ordered list of most recommended items
        """

There also exists a wrapper function for this to a purely list-based function.
This also takes care of the translation between Muzilla ID and the model's Index:

    def recommend_items_list(self, artist_ids: list[int], album_counts=None, topn: int = 10, excluded_artists=None) -> list[int]:

