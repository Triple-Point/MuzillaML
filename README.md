# MuzillaML
Machine Learning for the Muzilla project

## Usage
python main --config [path to config.yaml]

### config.yaml

    data:
        raw_data: [path to the latest Muzilla data]

    model:
        model: [cosine_model|tfidf_model|popularity_model|random_model]

## Project description

The goal of the project evaluate recommendation models for the Muzilla task.
That is given a list of users, artists and album counts, can we recommend a new artist that the user would be interested in?  


## Data description

The Muzilla data set is proprietary. It is formatted as user, artist and count of albums.

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

There are approx. 400k users, 500k artists, and 60M entries.

This format makes it trivial to read into a Sparce COO Matrix.

### Alternative datasets

- The Audioscrobbler data set contains profiles for users and a play count of artists that each user listens to.
  - Originalk dataset no longer publicly available 
  - [LastFM dataset on Kaggle](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)
- Spotify Million Playlist Dataset
  - No longer publicly available

## Evaluation Metrics

### Top-𝑘 masked recommendation:
 - Hide a portion of artists in the test set. Evaluate if the model can correctly predict these hidden interactions.
 - For each user in the test set generate top-𝑘 recommendations. 
 - Compare the recommendations with the ground truth (Masked items).

 - The code currently uses a default of 10

### Mean Average Precision (MAP)

Assuming order is important, MAP seems like a good metric to use.
The model generates the top-𝑘 and compares this to 𝑘 masked artists.

### Results

#### Baseline
The random_model generates a list of 𝑘 randomly selected artists.

10-fold cross validation resulted in MAP of [ 
2.07 
0.83
7.63
1.06
0.60
1.22
1.24
2.53
0.83
0.0] * 10^-6

...or basically zero.

#### Popularity model

The popularity model calculates the most popular artists, and recommends them.

Not optimised for time, so I stopped before cross validation was done. Converged on an average MAP score of: 0.002827381577617915
Better than chance. w00t!!!

#### Cosine Similarity model

The Cosine Similarity finds a user with the lowest cosine angle, and selects the most popular artists from that user. If there are not enough recommendations, the next closest user is selected and so on.

Using 10% of the data, and using the same data set for training and testing, the MAP converges to about 0.60

Cross validation results OTW

### Serendipity 

TDB. This is actually an important part of the Muzilla philosophy, but this is hard to define ;)

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

