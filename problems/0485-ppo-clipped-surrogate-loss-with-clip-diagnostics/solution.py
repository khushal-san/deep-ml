import numpy as np

def ppo_clip_loss(old_log_probs: np.ndarray, new_log_probs: np.ndarray, advantages: np.ndarray, epsilon: float = 0.2) -> tuple:
    """
    Compute the PPO clipped surrogate loss and clip fraction diagnostic.
    
    Args:
        old_log_probs: Log-probabilities of actions under the old policy, shape (N,)
        new_log_probs: Log-probabilities of actions under the new policy, shape (N,)
        advantages: Advantage estimates for each sample, shape (N,)
        epsilon: Clipping parameter for the probability ratio
    
    Returns:
        Tuple of (loss, clip_fraction), both rounded to 4 decimal places
        - loss: Mean negated clipped surrogate objective (for minimization)
        - clip_fraction: Fraction of samples where the ratio was clipped
    """
    log_probs_ratio = np.exp(new_log_probs - old_log_probs)
    clipped_log_probs_ratio = np.clip(log_probs_ratio, 1-epsilon, 1+epsilon)
    loss = - np.mean(clipped_log_probs_ratio * advantages)
    clip_fraction = np.sum(np.where(np.abs(log_probs_ratio - clipped_log_probs_ratio) > 0, 1, 0)) / old_log_probs.shape[0]
    return (loss, clip_fraction) 

