
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score

def estimate_transition_matrix(labels, K):
    """Estimate KxK row-stochastic transition matrix from a state sequence.

    Parameters
    ----------
    labels : array-like, shape (T,)
        Discrete state sequence in {0, ..., K-1}.
    K : int
        Number of states.

    Returns
    -------
    P_hat : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    """
    labels = np.asarray(labels, dtype=int)
    T = len(labels)
    P = np.zeros((K, K), dtype=float)
    if T <= 1:
        return np.full((K, K), 1.0 / K, dtype=float)
    for t in range(T - 1):
        i = labels[t]
        j = labels[t + 1]
        if 0 <= i < K and 0 <= j < K:
            P[i, j] += 1.0
    row_sums = P.sum(axis=1, keepdims=True)
    # avoid divide-by-zero
    row_sums[row_sums == 0] = 1.0
    return P / row_sums

def basin_recovery_ari(g_true, g_hat):
    """Basin recovery score as Adjusted Rand Index between true and inferred labels."""
    g_true = np.asarray(g_true, dtype=int)
    g_hat = np.asarray(g_hat, dtype=int)
    return float(adjusted_rand_score(g_true, g_hat))

def transition_matrix_agreement(P_true, P_hat):
    """Transition Matrix Agreement (TMA).

    We quantify similarity between two row-stochastic KxK matrices using
    1 - ||P_true - P_hat||_F / ||P_true||_F, clipped to [0, 1].

    Returns 1 when matrices are identical, and decreases as they diverge.
    """ 
    P_true = np.asarray(P_true, dtype=float)
    P_hat = np.asarray(P_hat, dtype=float)
    # basic sanity: normalize rows of P_hat in case caller forgot
    row_sums = P_hat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    P_hat = P_hat / row_sums
    diff = P_true - P_hat
    num = np.linalg.norm(diff)
    denom = np.linalg.norm(P_true) + 1e-12
    tma = 1.0 - num / denom
    return float(np.clip(tma, 0.0, 1.0))

def state_means_from_labels(X, labels, K):
    """Compute per-state means given time series X and labels.

    Parameters
    ----------
    X : ndarray, shape (T, D)
        Time series.
    labels : ndarray, shape (T,)
        State labels in {0, ..., K-1}.
    K : int
        Number of states.

    Returns
    -------
    means : ndarray, shape (K, D)
        Mean of X within each state. States that never occur get NaNs.
    """ 
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    T, D = X.shape
    means = np.full((K, D), np.nan, dtype=float)
    for k in range(K):
        mask = labels == k
        if np.any(mask):
            means[k] = X[mask].mean(axis=0)
    return means

def state_distance_alignment(means_true, means_hat):
    """State Distance Alignment (SDA).

    Computes Pearson correlation between vectorized pairwise distance
    matrices of true and inferred state centroids.

    Parameters
    ----------
    means_true : ndarray, shape (K, D)
    means_hat : ndarray, shape (K, D)

    Returns
    -------
    sda : float
        Correlation in [-1, 1]. NaN if fewer than 2 valid states.
    """ 
    means_true = np.asarray(means_true, dtype=float)
    means_hat = np.asarray(means_hat, dtype=float)
    # discard states that are all-NaN in either set
    valid = ~(np.isnan(means_true).all(axis=1) | np.isnan(means_hat).all(axis=1))
    if valid.sum() < 2:
        return float("nan")
    mt = means_true[valid]
    mh = means_hat[valid]
    Dt = squareform(pdist(mt))
    Dh = squareform(pdist(mh))
    v1 = Dt.ravel()
    v2 = Dh.ravel()
    v1 = v1 - v1.mean()
    v2 = v2 - v2.mean()
    num = np.dot(v1, v2)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-12
    if denom == 0:
        return float("nan")
    return float(num / denom)

def compute_metrics_all(g_true, g_hat, K, P_true=None, X=None):
    """Convenience wrapper to compute BR_ARI, TMA, SDA.

    Parameters
    ----------
    g_true : ndarray, shape (T,)
        Ground-truth state labels.
    g_hat : ndarray, shape (T,)
        Estimated labels.
    K : int
        Number of states.
    P_true : ndarray, optional, shape (K, K)
        Ground-truth transition matrix, if available.
    X : ndarray, optional, shape (T, D)
        Time series used to infer states. Required for SDA.

    Returns
    -------
    metrics : dict
        Keys: 'BR_ARI', 'TMA', 'SDA'.
    """ 
    g_true = np.asarray(g_true, dtype=int)
    g_hat = np.asarray(g_hat, dtype=int)

    br = basin_recovery_ari(g_true, g_hat)

    if P_true is not None:
        P_hat = estimate_transition_matrix(g_hat, K)
        tma = transition_matrix_agreement(P_true, P_hat)
    else:
        tma = float("nan")

    if X is not None:
        means_true = state_means_from_labels(X, g_true, K)
        means_hat = state_means_from_labels(X, g_hat, K)
        sda = state_distance_alignment(means_true, means_hat)
    else:
        sda = float("nan")

    return {"BR_ARI": br, "TMA": tma, "SDA": sda}
