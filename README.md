# MusillaML
Machine Learning for the Musilla project

## Usage
python main --config [path to config.yaml]

### config.yaml

    data:
        raw_data: [path to the latest Musilla data]

## Project description

The goal of the project is to recommend a new artist to an existing user, given their current collection of albums. 

## Data description

The Musilla data set is proprietary. It is formatted as user, artist and count of albums.

Where albums are credited to more than one artist 
(For example, [Split albums](https://en.wikipedia.org/wiki/Split_album) and [Compilation albums](https://en.wikipedia.org/wiki/Compilation_album)), 
the count incremented by a fraction of each artist's contribution. 
For example, the album _Short Music for Short People_ with 101 artists will add approximately 0.01 to each artist's album count.   

Excerpt below. Columns are UserID, ArtistID, AlbumCount:

    175772,371706,0.375
    175772,564687,0.3
    175772,254653,0.25
    175772,21457,1
    51815,429978,1
    51815,61121,2
    51815,882847,2

### Alternative datasets

- The Audioscrobbler data set contains profiles for users and a play count of artists that each user listens to.
  - Originalk dataset no longer publicly available 
  - [LastFM dataset on Kaggle](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)
- Spotify Million Playlist Dataset
  - No longer publicly available

## Evaluation Metrics

### Vectors similarity - nearest neighbour

If the user is represented as a binary vector (that is, the artist is considered either present or not) several metrics are possible, including:

- [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

Most of these are also suitable for (normalized) artist counts. 

### K-nearest neighbour

[K-nearest neighbour](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

### Artist Candidate Selection

Given a similar user (or multiple similar users) the candidate list can be constructed. 
No artist in the current user's collection should appear in the recommendations. 

### Artist Candidate Ranking

