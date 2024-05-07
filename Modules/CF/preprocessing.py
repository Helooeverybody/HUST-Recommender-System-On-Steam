import pandas as pd
import numpy as np
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


class PreProcessing:

    def get_mapper_and_shape(
        ratings_data: NDArray,
        test_data: NDArray | None = None,
        to_name: dict | None = None,
    ):
        if test_data is not None:
            all_users = np.unique(np.concatenate((ratings_data[:, 0], test_data[:, 0])))
            all_items = np.unique(np.concatenate((ratings_data[:, 1], test_data[:, 1])))
        else:
            all_users = np.unique(ratings_data[:, 0])
            all_items = np.unique(ratings_data[:, 1])
        num_users = len(all_users)
        num_items = len(all_items)
        shape = num_users, num_items
        M = Mapping(
            user_to_index=dict(zip(all_users, range(num_users))),
            index_to_user=dict(zip(range(num_users), all_users)),
            index_to_item=dict(zip(range(num_items), all_items)),
            item_to_index=dict(zip(all_items, range(num_items))),
            item_to_name=to_name,
        )
        return M, shape

    def data_from_matrix(train_data, test_data):
        M, shape = PreProcessing.get_mapper_and_shape(train_data, test_data)
        X_train = np.apply_along_axis(
            lambda x: (M.user_to_index[x[0]], M.item_to_index[x[1]]),
            axis=1,
            arr=train_data,
        )
        X_test = np.apply_along_axis(
            lambda x: (M.user_to_index[x[0]], M.item_to_index[x[1]]),
            axis=1,
            arr=test_data,
        )
        return X_train, X_test, shape

    def data_from_csv(fname: str) -> tuple[NDArray, NDArray, Mapping, tuple[int, int]]:
        """File structure: user_id, app_id, title, rating"""
        # Get data [user: 0, item: 1, name: 2, rating: 3]
        ratings_df = pd.read_csv(fname).to_numpy()
        # Get users and games listing
        all_users = np.unique(ratings_df[:, 0])
        all_items = np.unique(ratings_df[:, 1])
        num_users = len(all_users)
        num_items = len(all_items)
        item_to_name = dict(zip(ratings_df[:, 1], ratings_df[:, 2]))
        # Get mapper and ui matrix shape
        M, ui_shape = PreProcessing.get_mapper_and_shape(
            ratings_df[:, [0, 1]], to_name=item_to_name
        )
        ui_shape = (num_users, num_items)
        X = np.apply_along_axis(
            lambda x: (M.user_to_index[x[0]], M.item_to_index[x[1]]),
            axis=1,
            arr=ratings_df[:, [0, 1]],
        )
        y = ratings_df[:, 3].astype("float")

        return X, y, M, ui_shape

    def split(X, y, train_proportion: float, shuffle=False) -> tuple[NDArray]:
        data_size = len(X)
        idx = np.arange(data_size)
        if shuffle:
            np.random.shuffle(idx)
        train_idx = idx[: int(train_proportion * data_size)]
        test_idx = idx[int(train_proportion * data_size) :]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X, y, mapping, shape = PreProcessing.data_from_csv("./data/test_data.csv")
    print(X, y)
    X_train, y_train, X_test, y_test = PreProcessing.split(X, y, 0.8, shuffle=True)
    print(f"Data count:  {len(X)}")
    print(f"Train count: {len(X_train)}")
    print(f"Test count:  {len(X_test)}")
    print(f"Matrix shape: {shape}")
