from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import src
import numpy as np

def test_euclidean_distances():
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = euclidean_distances(x, y)
    _est = src.euclidean_distances(x, y)
    assert (np.allclose(_true, _est))


def test_manhattan_distances():
    x = np.random.rand(100, 100)
    y = np.random.rand(100, 100)
    _true = manhattan_distances(x, y)
    _est = src.manhattan_distances(x, y)
    assert (np.allclose(_true, _est))
