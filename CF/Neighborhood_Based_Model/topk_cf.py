from typing import Callable, Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Typing
NDArray = np.ndarray
DataFrame = pd.DataFrame


class TopKNeighborCF:

    def __init__(
        self,
        sim_func: Callable,
        mode: Literal["uucf", "iicf"] = "uucf",
        neighbors=5,
        normalize=True,
    ) -> None:
        # System setting
        self.mode = mode if mode == "iicf" else "uucf"
        self.dist_func = sim_func
        self.n_neighbors = neighbors
        self.normalize = normalize
        # Matrix data
        self.X: NDArray
        self.y: NDArray
        # Other attributes
        self.mu = NDArray | None
        self.ui_matrix: csr_matrix
        self.sim_matrix: NDArray
        # Users and items count
        self.num_users: int
        self.num_items: int

    def fit(self, X: NDArray, y: NDArray, user_item_shape: tuple[int, int]) -> None:
        """
        Fit data into model.
        ### Parameters:
        - X: matrix of users and items
        - y: corresponding ratings
        - user_item_shape: (num_users, num_items)
        """
        # Get data, users count, items count
        self.X = X
        self.y = y.astype("float")
        self.num_users = user_item_shape[0]
        self.num_items = user_item_shape[1]
        self._build()

    def _get_user_item_matrix(self) -> csr_matrix:
        """Return the sparse matrix representation of the user-item matrix."""
        user_index = self.X[:, 0]
        item_index = self.X[:, 1]
        user_item_matrix = csr_matrix(
            (self.y, (item_index, user_index)),
            shape=(self.num_items, self.num_users),
        )
        return user_item_matrix

    def _get_similarity_matrix(self) -> NDArray:
        """Return the similarity matrix calculated from the sparse user-item matrix."""
        if self.mode == "iicf":
            return self.dist_func(self.ui_matrix)
        else:  # self.mode == "uucf"
            return self.dist_func(self.ui_matrix.T)

    def _normalize_data(self) -> None:
        """Normalize the data"""
        if self.mode == "iicf":
            col = self.X[:, 1].astype("int")
            all_values = range(self.num_items)
        else:  # self.mode == "uucf"
            col = self.X[:, 0].astype("int")
            all_values = range(self.num_users)
        self.mu = np.zeros(len(all_values))

        rating = self.y
        for v in all_values:
            # Get all non empty position
            idx = np.where(col == v)[0]
            # Calculate the mean of by user or item
            if len(idx) == 0:
                mean = 0.0
            else:
                mean = rating[idx].mean()
            self.mu[v] = mean
            # Normalize by subtracting the mean
            for i in idx:
                self.y[i] -= mean

    def _build(self):
        """Build the CF system"""
        # Normalize the data
        if self.normalize:
            self._normalize_data()
        # Get user-item matrix for iicf or utem-user matrix for uucf
        self.ui_matrix = self._get_user_item_matrix()
        # Get similarity matrix
        self.sim_matrix = self._get_similarity_matrix()

    def get_rating_prediction(self, user_idx, item_idx) -> float:
        """Return the predicted rating of an user for an item."""
        if self.mode == "iicf":
            idx = item_idx
            # Get all games that user played
            also_played = self.ui_matrix.T.getrow(user_idx).indices
            ratings = self.ui_matrix.T.getrow(user_idx).data
        else:  # self.mode == "uucf"
            idx = user_idx
            # Get all user that also played the game
            also_played = self.ui_matrix.getrow(item_idx).indices
            ratings = self.ui_matrix.getrow(item_idx).data

        # Get sim scores
        sim_scores = self.sim_matrix[idx][also_played]
        # Sort by sim scores
        neighbours_idx = np.argsort(sim_scores)[-self.n_neighbors :]
        sim_scores = sim_scores[neighbours_idx]
        ratings = ratings[neighbours_idx]

        # Calculate prediction
        pred = np.sum(ratings * sim_scores) / (abs(sim_scores).sum() + 1e-8)
        if self.normalize:
            pred += self.mu[idx]
        return pred

    def predict(self, X_test: NDArray):
        return np.apply_along_axis(
            lambda x: self.get_rating_prediction(x[0], x[1]), 1, X_test
        )

    # Get recommendation for an user
    def get_recommendation(self, user_idx: int, n_recommnedation: int = 10) -> NDArray:
        """Return a array of n games that the user has not played and their
        corresponding predicted ratings."""

        # Games that user has played
        user_played = self.ui_matrix.T.getrow(user_idx).indices
        # Games that user has not played (to be recommended)
        user_not_played = np.array(
            [i for i in range(self.num_items) if i not in user_played]
        )
        # Calculate the predicted ratings for the non-played games
        ratings = np.array(
            [
                self.get_rating_prediction(user_idx, game_idx)
                for game_idx in user_not_played
            ]
        )
        idx = np.flip(np.argsort(ratings))[:n_recommnedation]
        return user_not_played[idx], ratings[idx]
