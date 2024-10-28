# Artist Recommender System

## Dataset

Original dataste can be foundd [here](https://grouplens.org/datasets/hetrec-2011/).
However, for convenience, I have uploaded just the files that I used in this project: 

- artists.dat: matches artist id to name and other information (not relavant for this project)
- user_artists.dat: matches user id to artist id and weight (representing the preference of the user)

## Model

This basic music recommender system uses the library [Implicit](https://github.com/benfred/implicit),
more specifically, the Alternating Least Squares model (Collaborative Filtering for Implicit Feedback Datasets).

This model takes as input an Sparse Matrix (CSR) of user-artists interactions.

## Usage

Uses [uv](https://docs.astral.sh/uv/) to describe project dependencies.ç


## Future Work

Be able to access Spotify's data to get the user-artists interactions.
