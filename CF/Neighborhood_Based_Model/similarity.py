import numpy as np
from sklearn.metrics import pairwise
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
from typing import Callable, Literal

NDArray = np.ndarray


class Similarity:
    """Methods that return the similarity matrix for CF's calculation."""

    def cosine(matrix: csr_matrix) -> NDArray:
        """Return the cosine similarity matrix."""
        return pairwise.cosine_similarity(matrix)

    def from_distance(
        metric: Literal["cosine", "euclidean", "manhattan", "haversine"] = "cosine",
        tosim: Callable = lambda x: 1 / (1 + x),
    ) -> Callable:
        """Return a similarity matrix by converting a distance matrix's calculation."""

        def inner(matrix: csr_matrix) -> NDArray:
            distance_matrix = pairwise.distance_metrics()[metric](matrix)
            sim_matrix = tosim(distance_matrix)
            return sim_matrix

        return inner

    def inverse_euclidean_squared(matrix: csr_matrix) -> NDArray:
        return Similarity.from_distance("euclidean", lambda x: 1 / (1 + x**2))(matrix)

    def pearson(matrix: csr_matrix) -> NDArray:
        """Return the Pearson similarity matrix."""
        matrix = matrix.toarray()
        sim_matrix = np.corrcoef(matrix)
        return np.nan_to_num(sim_matrix, copy=False)

    def spearman(matrix: csr_matrix) -> NDArray:
        """Return the Spearman similarity matrix."""
        matrix = matrix.toarray()
        sim_matrix = spearmanr(matrix)[0]
        return np.nan_to_num(sim_matrix, copy=False)
