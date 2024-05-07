import os
import pickle as pkl
import numpy as np
import pandas as pd
from typing import Callable
from scipy.sparse import csr_matrix, save_npz, load_npz
from collections import namedtuple

DataFrame = pd.DataFrame
NDArray = np.ndarray
Mapping = namedtuple(
    "Mapping",
    [
        "user_to_index",
        "index_to_user",
        "index_to_item",
        "item_to_index",
        "item_to_name",
    ],
)


class CF:
    def __init__(
        self,
        sim_func: Callable,
        mode: str = "uucf",
        n_neighbors=5,
        normalize=True,
        save_cache=False,
    ) -> None:
        # System setting
        self.mode = mode if mode == "iicf" else "uucf"
        self.dist_func = sim_func
        self.n_neighbors = n_neighbors
        self.normalize = normalize
        self.save = save_cache
        # Matrix data
        self.X: NDArray
        self.y: NDArray
        # self.data = NDArray
        self.mu = NDArray
        self.ui_matrix: csr_matrix
        self.sim_matrix: NDArray
        # Users and items count
        self.num_users: int
        self.num_items: int
        # Cache location
        self.cache_dir = "./Modules/CF/cache"

    def set_cache_dir(self, cache_dir: str):
        """ "Set cache location"""
        self.cache_dir = cache_dir

    def _get_user_item_matrix(self) -> csr_matrix:
        """Return the sparse matrix representation of the user-item matrix"""
        user_index = self.X[:, 0]
        item_index = self.X[:, 1]
        user_item_matrix = csr_matrix(
            (self.y, (item_index, user_index)),
            shape=(self.num_items, self.num_users),
        )
        return user_item_matrix

    def _get_similarity_matrix(self, ui_matrix: csr_matrix):
        if self.mode == "iicf":
            return self.dist_func(ui_matrix)
        else:  # self.mode == "uucf"
            return self.dist_func(ui_matrix.T)

    def _normalize_data(self) -> None:
        if self.mode == "iicf":
            col = self.X[:, 1].astype("int")
            all_values = range(self.num_items)
        else:  # self.mode == "uucf"
            col = self.X[:, 0].astype("int")
            all_values = range(self.num_users)
        self.mu = np.zeros(len(all_values))
        if not self.normalize:
            return

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

    def fit(self, X: NDArray, y: NDArray, user_item_shape: tuple[int, int]) -> None:
        """
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

    def _build(self):
        """Build the CF system"""
        # Normalize the data
        self._normalize_data()
        # Get user-item matrix for iicf or utem-user matrix for uucf
        ui_matrix = self._get_user_item_matrix()
        self.ui_matrix = ui_matrix
        # Get similarity matrix
        self.sim_matrix = self._get_similarity_matrix(ui_matrix)
        # Save cache
        if self.save:
            self.save_cache()

    # Handling cache data
    def save_cache(self):
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        with open(f"{self.cache_dir}/system.pkl", "wb") as file:
            system_dict = {
                "mode": self.mode,
                "k": self.n_neighbors,
                "users": self.num_users,
                "games": self.num_items,
                "normed": self.normalize,
            }
            pkl.dump(system_dict, file)
        save_npz(f"{self.cache_dir}/ui_matrix.npz", self.ui_matrix)
        np.save(f"{self.cache_dir}/mu", self.mu)
        np.save(f"{self.cache_dir}/sim_matrix", self.sim_matrix)

    def load_cache(self):
        with open(f"{self.cache_dir}/system.pkl", "rb") as data:
            system_dict: dict = pkl.load(data)
            self.mode = system_dict["mode"]
            self.n_neighbors = system_dict["k"]
            self.num_users = system_dict["users"]
            self.num_items = system_dict["games"]
            self.normalize = system_dict["normed"]
        self.ui_matrix = load_npz(f"{self.cache_dir}/ui_matrix.npz")
        self.mu = np.load(f"{self.cache_dir}/mu.npy")
        self.sim_matrix = np.load(f"{self.cache_dir}/sim_matrix.npy")

    # Get prediction
    def get_rating_prediction(self, user_idx, item_idx) -> float:
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

    def predict(self, X):
        return np.apply_along_axis(
            lambda x: self.get_rating_prediction(x[0], x[1]), 1, X
        )

    # Get recommendation for an user
    def get_recommendation(self, user_idx: int, n_recommnedation: int = 10) -> NDArray:
        # Games that user has played
        user_played = self.ui_matrix.T.getrow(user_idx).indices
        # Games that user has not played (to be recommended)
        user_not_played = np.array(
            [i for i in range(self.num_items) if i not in user_played]
        )
        # Calculate the predicted ratings for the not-played games
        ratings = np.array(
            [
                self.get_rating_prediction(user_idx, game_idx)
                for game_idx in user_not_played
            ]
        )
        idx = np.flip(np.argsort(ratings))[:n_recommnedation]
        return user_not_played[idx], ratings[idx]
