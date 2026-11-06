import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    X_batch = X[batch_indices, :]
    y_batch = y[batch_indices]

    y_pred = X_batch @ weights + bias
    dL_dw = (2 / X_batch.shape[0]) * (X_batch.T @ (y_pred - y_batch))
    dL_db = (2 / X_batch.shape[0]) * np.sum(y_pred - y_batch)

    weights = weights - lr * dL_dw
    bias = bias - lr * dL_db

    weights = weights.tolist()
    weights.append(bias)

    return weights
