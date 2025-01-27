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

## Models

### Baseline - Random recommendations
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
0.0] * 10<sup>-6</sup>

...or basically zero.

### Popularity model

The popularity model calculates the most popular artists, and recommends them.

Cross-validation MAP scores of [
7.05, 6.79, 7.09, 6.94, 6.85, 7.13, 6.94, 7.05, 7.02, 6.68] * 10<sup>-3</sup>

Average MAP score = 6.96 * 10<sup>-3</sup>

Better than chance. w00t!!!

### Cosine Similarity model

#### Vectors similarity - nearest neighbour

Several metrics are possible, including:

- [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
- [Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)

#### K-nearest neighbour

[K-nearest neighbour](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

The Cosine Similarity model finds a user with the lowest cosine angle, and selects the most popular artists from that user. If there are not enough recommendations, the next closest user is selected and so on.

Using 10% of the data, and using the same data set for training and testing, the MAP converges to about 0.60

Cross validation results 0.10

#### Artist Candidate Selection

Given a similar user (or multiple similar users) the candidate list can be constructed. 
No artist in the current user's collection should appear in the recommendations. 

#### Artist Candidate Ranking

The most popular artist is ranked higher.  

## Results

| Model               | MAP@10 |
|---------------------|-------|
| Random Selection    | 0.00  |
| Popularity Matching | 0.01  |
| Cosine Similarity   | 0.10  | 

### Serendipity 

TBD. This is actually an important part of the Muzilla philosophy, but this is hard to define ;)
