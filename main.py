from pathlib import Path
from typing import Optional

import implicit
import implicit.recommender_base
import pandas as pd
from scipy.sparse import csr_matrix


USER_ARTISTS_PATH = Path("lastfm_data/user_artists.dat")
ARTISTS_PATH = Path("lastfm_data/artists.dat")


def load_user_artists() -> pd.DataFrame:
    """Load the user-artists data from the file and return it as a sparse matrix and a dataframe."""
    df = pd.read_csv(USER_ARTISTS_PATH, sep="\t")

    # set a composed index using the user ID and artist ID
    df.set_index(["userID", "artistID"], inplace=True)

    # convert the wieght column to a float
    df["weight"] = df["weight"].astype(float)

    return df

def load_artists() -> pd.DataFrame:
    """Load the artists data from the file and return it as a dataframe."""
    df = pd.read_csv(ARTISTS_PATH, sep="\t")

    return df


def create_sparse_matrix_from_dataframe(df: pd.DataFrame) -> csr_matrix:
    """Create a sparse matrix from a dataframe."""

    # fit the datafram into a sparse matrix
    csr = csr_matrix(
        (
            df["weight"], (
                df.index.get_level_values("userID"), 
                df.index.get_level_values("artistID")
            )),
    )

    return csr


def retrieve_artist_name(artist_id: int, df: Optional[pd.DataFrame] = None) -> str:
    """Retrieve the name of an artist from the artists dataframe."""
    if not df:
        df = load_artists()

    return df.loc[artist_id, "name"]


class ImplicitRecommender:
    def __init__(
            self, 
            user_artists_df: pd.DataFrame, 
            artists_df: pd.DataFrame,
            model: implicit.recommender_base.RecommenderBase
        ):
        self.user_artists_df = user_artists_df
        self.artists_df = artists_df
        self.model = model

    def fit(self, user_arrtists_matrix: csr_matrix):
        self.model.fit(user_arrtists_matrix)

    def recommend(self, user_id: int,user_artists_matrix: csr_matrix, n: int = 5) -> list[tuple[str, float]]:
        """
        Recommend the top artists for a user.

        Parameters:
        user_id (int): The ID of the user to recommend for.
        top_artits (int): The number of top artists to recommend.

        Returns:
        """

        artists_id, scores = self.model.recommend(user_id, user_artists_matrix[user_id], N=n)
        top_artists = [(retrieve_artist_name(artist_id), score) for artist_id, score in zip(artists_id, scores)]
        return top_artists


def main():
    user_artists_df = load_user_artists()
    artists_df = load_artists()

    # create a sparse matrix from the dataframe
    user_artists_matrix = create_sparse_matrix_from_dataframe(user_artists_df)

    # init model
    model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01)
    recommender = ImplicitRecommender(user_artists_df, artists_df, model)
    recommender.fit(user_artists_matrix)

    # recommend the top artists for a user
    top_artists = recommender.recommend(1, user_artists_matrix, 5)

    print(top_artists)


if __name__ == "__main__":
    main()
