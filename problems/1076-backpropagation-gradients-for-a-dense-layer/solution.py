import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute gradients of the squared-error cost w.r.t. W, b, and a_prev
    for a dense layer z = W @ a_prev + b followed by sigmoid activation.

    Args:
        a_prev: activations from previous layer, shape (K,)
        W: weight matrix, shape (J, K)
        b: bias vector, shape (J,)
        y: target vector, shape (J,)

    Returns:
        dict with keys 'dW' (J x K nested list), 'db' (list of length J),
        and 'da_prev' (list of length K).
    """
    # Your code here
    logits = W @ a_prev + b
    y_pred = 1/(1+np.exp(-logits))
    dW = (2 * (y_pred - y) * y_pred * (1-y_pred)).reshape(-1, 1) @ a_prev.reshape(1, -1)
    db = (2 * (y_pred - y) * y_pred * (1-y_pred))
    da_prev = W.T @ (2 * (y_pred - y) * y_pred * (1-y_pred))

    return {
        "dW": dW.tolist(),
        "db": db.tolist(),
        "da_prev": da_prev.tolist(),
    }

