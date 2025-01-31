from src.data.FileUtils import ArtistLookup
from src.models.model_factory import model_factory


class WebRecommender:
    def __init__(self, artist_lookup_table: str, model_class:str, raw_data: str, processed_data: str):
        self.artist_name_lookup = ArtistLookup(artist_lookup_table)
        model_class = model_factory(model_class)
        self.model = model_class.from_files(raw_data, processed_data)

    def recommend_items(self, artist_list: list[int], album_counts: list[int], num: int, excluded_artists:list[int]) -> list[str]:
        print(f"{artist_list=}")
        recommended_artists = self.model.recommend_items_list(artist_list, album_counts, num, excluded_artists)
        artist_strings = self.artist_name_lookup.ids_to_artists(recommended_artists)
        print(f"{artist_strings=}")
        return artist_strings
