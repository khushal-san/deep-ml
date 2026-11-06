import numpy as np

def compute_cross_entropy_loss(predicted_probs, true_labels, epsilon=1e-15):
    return -np.mean(
        np.sum(true_labels * np.log(predicted_probs + epsilon), axis=1)
    )