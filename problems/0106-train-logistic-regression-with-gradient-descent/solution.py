import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	# Your code here
	m, n = X.shape
	y = y.reshape(-1, 1)
	weights = np.zeros((n, 1))
	bias = np.zeros((1, 1))

	losses = []

	for iter in range(iterations):
		logits = X @ weights + bias 
		preds = 1/(1+np.exp(-logits))
		loss =  - np.sum(y * np.log(preds) + (1-y) * np.log((1 - preds)))
		losses.append(loss)
		dL_dweights = X.T @ (preds - y)
		dL_dbias = 	np.sum((preds - y))
		weights = weights - learning_rate * dL_dweights
		bias = bias - learning_rate * dL_dbias

	coeffs = bias.tolist()[0]
	coeffs.extend(weights.flatten().tolist())
	return coeffs, losses 
