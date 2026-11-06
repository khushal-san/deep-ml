import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	# Your code here
	return gamma * ((X - np.mean(X, axis = -1, keepdims = True))/np.std(X, axis = -1, keepdims = True)) + beta