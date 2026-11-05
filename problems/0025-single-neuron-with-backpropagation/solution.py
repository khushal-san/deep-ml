import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	
	m, n = features.shape
	initial_weights = initial_weights.reshape(-1, 1)
	initial_bias = np.array(initial_bias).reshape(1, 1)
	labels = labels.reshape(-1, 1)

	updated_weights = initial_weights
	updated_bias = initial_bias
	mse_values = []

	for epoch in range(epochs):
		logits = features @ updated_weights  + updated_bias
		y_pred = 1/(1+np.exp(-logits))
		loss = np.mean((y_pred - labels) ** 2, axis = 0)
		mse_values.append(loss.item())
		dL_dw = (2/m) * (features.T @ ((y_pred - labels) * y_pred * (1-y_pred)))
		dL_db = 2 * np.mean(((y_pred - labels) * y_pred * (1-y_pred)), axis = 0)
		updated_weights = updated_weights - learning_rate * dL_dw
		updated_bias = updated_bias - learning_rate * dL_db

	return updated_weights.flatten(), updated_bias.flatten()[0], mse_values