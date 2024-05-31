import sys
import os
import json
from typing import Literal

import pandas as pd
import numpy as np

# Typing
DataFrame = pd.DataFrame
NDArray = np.ndarray

# Settings
working_path = os.path.abspath(".")
index = os.path.abspath(".").find("SourceCode")
if index >= 0:
    data_folder = os.path.join(working_path[:index], "data")
else:
    data_folder = os.path.join(working_path, "data")

source_folder = os.path.join(data_folder, "kaggle")
hours_intervals = [0.0, 2.0, 6.0, 14.1, 39.7]  # INF
ratings = [2, 2.5, 3, 3.5, 4]


class Preprocessor:
    """Collection of preprocess functions."""

    def assign_rating(hours: float, is_recommended: bool) -> float:
        """Helper function to assign rating
        based on hours played and implicit ratings.
        ### Returns:
            float rating
                In the range 1 to 5
        """
        for start, rating in zip(hours_intervals, ratings):
            if start <= hours:
                from_hour = rating
            else:
                break
        return from_hour + 2 * is_recommended - 1

    def generate_data_folder(review_limit: int, folder: str) -> None:
        """Extract data from source based on review limit into a folder."""
        assert isinstance(review_limit, int)
        # Create directory if not exist
        if not os.path.isdir(folder):
            os.makedirs(folder)

        # Get data from source
        print("Loading data from source ", end=" ")
        recommendations = pd.read_csv(
            os.path.join(source_folder, "recommendations.csv")
        )
        games = pd.read_csv(os.path.join(source_folder, "games.csv"))
        users = pd.read_csv(os.path.join(source_folder, "users.csv"))
        print("Done")

        # Filter recommendation
        print("Filtering recommendations", end=" ")
        user_review_count = recommendations["user_id"].value_counts()
        valid_users = user_review_count[user_review_count >= review_limit].index
        # Filter user that has low review count
        f_rec = recommendations[
            recommendations["user_id"].isin(valid_users)
        ].reset_index(drop=True)
        f_rec.to_csv(folder + "/recommendations.csv")
        print("Done")

        # Filter users
        print("Filtering users          ", end=" ")
        f_users = users[users["user_id"].isin(valid_users)]
        f_users.to_csv(folder + "/users.csv")
        print(f"Done -> {len(f_users)} users")

        # Filter games
        print("Filtering games          ", end=" ")
        valid_games = f_rec["app_id"].unique()
        f_games = games[games["app_id"].isin(valid_games)]
        f_games.to_csv(folder + "/games.csv")
        print(f"Done -> {len(f_games)} games")

        # Filter metadata
        print("Filtering metadata       ", end=" ")
        with open(
            os.path.join(source_folder, "games_metadata.json"), "r", encoding="utf8"
        ) as data:
            line_count = 0
            with open(folder + "/games_metadata.json", "w", encoding="utf8") as file:
                for line in data:
                    metadict: dict = json.loads(line)
                    if metadict["app_id"] in valid_games:
                        file.write(line)
                        line_count += 1
        print(f"Done -> {line_count} dicts")

    def generate_user_item_ratings(folder: str):
        """Generate user-item ratings data and mapping from a data folder for CF."""
        recs = pd.read_csv(folder + "/recommendations.csv")[
            ["user_id", "app_id", "hours", "is_recommended"]
        ]
        games = pd.read_csv(folder + "/games.csv")

        # Create mapping
        all_users = [int(user) for user in recs["user_id"].unique()]
        all_items = [int(item) for item in recs["app_id"].unique()]
        num_users = recs["user_id"].nunique()
        num_items = recs["app_id"].nunique()
        print("Generating mapping       ", end=" ")
        mapping = {
            "user_to_index": dict(zip(all_users, range(num_users))),
            "index_to_user": dict(zip(range(num_users), all_users)),
            "item_to_index": dict(zip(all_items, range(num_items))),
            "index_to_item": dict(zip(range(num_items), all_items)),
            "item_to_name": dict(zip(games["app_id"], games["title"])),
        }
        with open(folder + "/mapping.json", "w") as file:
            json.dump(mapping, file)
        print("Done")

        # Create data
        print("Generating data          ", end=" ")
        users = [mapping["user_to_index"][int(user)] for user in recs["user_id"]]
        items = [mapping["item_to_index"][int(item)] for item in recs["app_id"]]
        ratings = [
            Preprocessor.assign_rating(hours, is_recommended)
            for (hours, is_recommended) in zip(recs["hours"], recs["is_recommended"])
        ]
        data = np.concatenate([[users], [items], [ratings]], axis=0, dtype="object").T
        np.savetxt(folder + "/user_item_data.txt", data, fmt="%s")
        print("Done")

    def load_data(fname: str) -> tuple[NDArray, NDArray, tuple[int, int]]:
        """Load data from file using numpy load function.
        ### Return:
        X: user-item pairs
        y: ratings
        ui_shape: shape of the user-item matrix"""
        data = np.loadtxt(fname, dtype="object")
        X = data[:, :2].astype("int")
        y = data[:, 2].astype("float").T
        num_user = len(np.unique(X[:, 0]))
        num_item = len(np.unique(X[:, 1]))
        return X, y, (num_user, num_item)

    def split(
        X: NDArray,
        y: NDArray,
        train_proportion: float = 0.8,
        by=Literal["user", "item"] | None,
        shuffle=False,
    ) -> tuple[NDArray]:
        """Split data equally based on users (or items) into train and test data."""
        data_size = len(X)
        index = np.arange(data_size)
        if by is None:
            if shuffle:
                np.random.shuffle(index)
            train_idx = index[: int(train_proportion * data_size)]
            test_idx = index[int(train_proportion * data_size) :]
        else:
            k = 0 if by == "user" else 1
            train_idx = np.array([], dtype="int64")
            test_idx = np.array([], dtype="int64")
            for i in np.unique(X[:, k]):
                idx = index[X[:, k] == i]
                if shuffle:
                    np.random.shuffle(idx)
                at = int(np.ceil(train_proportion * len(idx)))
                train_idx = np.concatenate((train_idx, idx[:at]))
                test_idx = np.concatenate((test_idx, idx[at:]))

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    limit: int = 40
    args = sys.argv
    if len(args) == 2:
        limit = int(args[1])
    folder: str = os.path.join(data_folder, f"steam_{limit}")
    Preprocessor.generate_data_folder(limit, folder)
    Preprocessor.generate_user_item_ratings(folder)
