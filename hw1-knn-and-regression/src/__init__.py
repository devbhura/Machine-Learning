from .k_nearest_neighbor import KNearestNeighbor
from .load_json_data import load_json_data
from .distances import euclidean_distances, manhattan_distances

from .metrics import mean_squared_error
from .polynomial_regression import PolynomialRegression
from .generate_regression_data import generate_regression_data
from .load_json_data import load_json_data

# Fix for determinism.
import numpy as np 
np.random.seed(0)