import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

def fit_exact(X, max_iter=20000, alpha=0.05, tol_mean=1e-4, tol_corr=1e-4, verbose=1000):
    X = 2*X - 1
    n, k = X.shape
    X_all = gen_all_state(n)
    X2_all = 2*X_all - 1
    h = np.zeros(n); W = np.zeros((n,n))
    X_mean = X.mean(axis=1)
    X_corr = (X @ X.T) / k; np.fill_diagonal(X_corr, 0)

    err_m = err_c = np.inf
    for t in range(max_iter):
        p = calc_prob(h, W, X_all)            # size 2^n
        Y_mean = X2_all @ p                   # E[s]
        # E[ s s^T ] fast (no diag(p)):
        Y_corr = (X2_all * p) @ X2_all.T
        np.fill_diagonal(Y_corr, 0)

        h += alpha * (X_mean - Y_mean)
        W += alpha * (X_corr - Y_corr)

        err_m = np.linalg.norm(X_mean - Y_mean) / n
        err_c = np.linalg.norm(X_corr - Y_corr, 'fro') / (n*(n-1))

        if verbose and (t % verbose == 0 or t == max_iter-1):
            print(f"[exact] it={t:6d} err_mean={err_m:.3e} err_corr={err_c:.3e}")

        if err_m < tol_mean and err_c < tol_corr:
            break

    return h, W, {"converged": (err_m < tol_mean and err_c < tol_corr),
                 "n_iter": t+1, "err_mean": err_m, "err_corr": err_c}

import numpy as np

def fit_exact_mem(X01, max_iter=20000, alpha=0.2, tol=1e-5,
                  print_every=100, anneal=0.98, clip=0.05, seed=0):
    """
    Exact Ising (MEM) fit by moment matching using all 2^D states.
    X01: (D, T) in {0,1}; returns h, W for ±1 spins.
    """
    rng = np.random.default_rng(seed)

    # data moments (±1 spins)
    S = 2*X01 - 1                         # (D,T)
    D, T = S.shape
    X_mean = S.mean(axis=1)               # (D,)
    X_corr = (S @ S.T)/T                  # (D,D)
    np.fill_diagonal(X_corr, 0.0)

    # all states
    X_all = np.array([list(f"{i:0{D}b}") for i in range(2**D)], dtype=int).T  # (D, 2^D)
    Z = 2*X_all - 1                       # (D, 2^D) in {±1}

    # params
    h = np.zeros(D)
    W = np.zeros((D, D))

    def model_moments(h, W):
        # E(z) = -0.5 z^T W z - h^T z  (z columns)
        Ez = -0.5*np.sum(Z * (W @ Z), axis=0) - (h @ Z)
        Ez -= Ez.min()  # stability
        p = np.exp(-Ez)
        p /= p.sum()

        Y_mean = Z @ p                               # (D,)
        Y_corr = (Z * p) @ Z.T                       # (D,D) == Z @ diag(p) @ Z.T
        np.fill_diagonal(Y_corr, 0.0)
        return Y_mean, Y_corr, p

    a = float(alpha)
    best = {"err": np.inf, "h": h.copy(), "W": W.copy()}
    for it in range(1, max_iter+1):
        Y_mean, Y_corr, p = model_moments(h, W)

        # moment residuals
        m_err = X_mean - Y_mean
        c_err = X_corr - Y_corr

        # simple clipped gradient step
        dh = np.clip(a * m_err, -clip, clip)
        dW = np.clip(a * c_err, -clip, clip)

        h += dh
        W += dW
        # keep W symmetric with zero diagonal
        W = 0.5*(W + W.T)
        np.fill_diagonal(W, 0.0)

        # track best by max-abs residual (safer than allclose)
        cur_err = max(np.abs(m_err).max(), np.abs(c_err).max())
        if cur_err < best["err"]:
            best = {"err": cur_err, "h": h.copy(), "W": W.copy()}

        if it % print_every == 0:
            print(f"[MEM] it={it:5d} max|mom-res|={cur_err:.3e}, step={a:.3g}")
            a *= anneal

        if cur_err < tol:
            print(f"[MEM] converged at it={it} with max|mom-res|={cur_err:.3e}")
            break
    else:
        # not converged
        print(f"[MEM] hit max_iter={max_iter}, best max|mom-res|={best['err']:.3e}")

    return best["h"], best["W"]


def load_testdata(n=1):
    data_file_name = f'./testdata_{n}.dat'
    roi_file_name = './roiname.dat'
    X = np.loadtxt(data_file_name, dtype=int)
    with open(roi_file_name, 'r') as f:
        roi_names = [line.strip() for line in f.readlines()]
    return (X == 1).astype(int)

# Helper functions
def binarize(X):
    return (X - X.mean(axis=1, keepdims=True) >= 0).astype(int)

def calc_state_no(X):
    """
    Vectorized binary-to-integer state index.
    X: (n, k) with entries {0,1}
    Returns shape (k,) ints in [0, 2^n - 1]
    """
    n = X.shape[0]
    weights = (1 << np.arange(n-1, -1, -1)).astype(np.int64)
    return X.T.dot(weights).astype(np.int64)


def gen_all_state(n):
    """Return all 2^n binary states as (n, S) with S=2^n."""
    S = 1 << n
    # generate integers [0..2^n-1], unpack bits
    ints = np.arange(S, dtype=np.uint32)
    bits = ((ints[:, None] >> np.arange(n-1, -1, -1)) & 1).astype(np.int8)  # (S, n)
    return bits.T  # (n, S)

def calc_energy(h, W, X):
    X = 2 * X - 1  # Convert to {+1, -1}
    return -0.5 * np.sum(X * W.dot(X), axis=0) - h.dot(X)

def calc_prob(h, W, X):
    energy = calc_energy(h, W, X)
    energy -= energy.min()  # Avoid overflow
    prob = np.exp(-energy)
    return prob / prob.sum()
    

def calc_accuracy(h, W, X):
    """
    Compare observed state distribution p_n to:
      - independent model p_1 (product of marginals)
      - Ising model p_2 (with h, W)
    Returns acc1 (entropy-based) and acc2 (divergence-based).
    """
    # observed over time
    state_no_obs = calc_state_no(X)                     # (T,)
    uniq_states, freq = np.unique(state_no_obs, return_counts=True)
    p_n = freq / freq.sum()

    # all states once
    X_all = gen_all_state(X.shape[0])                   # (n, S)
    S = X_all.shape[1]
    # indices of observed states in full space (vectorized)
    state_no_all = calc_state_no(X_all)                 # (S,)
    idx_map = {s: i for i, s in enumerate(state_no_all)}
    observed_idx = np.array([idx_map[s] for s in uniq_states])

    # independent model from marginals q (on 0/1)
    q = X.mean(axis=1)                                  # (n,)
    p_1 = np.prod(X_all * q[:, None] + (1 - X_all) * (1 - q[:, None]), axis=0)

    # ising model probs over all states
    p_2 = calc_prob(h, W, X_all)

    # map to observed support
    p_1_m = p_1[observed_idx]
    p_2_m = p_2[observed_idx]

    def entropy(p):
        p = np.clip(p, 1e-12, None)
        return -np.sum(p * np.log2(p))

    acc1 = (entropy(p_1) - entropy(p_2)) / (entropy(p_1) - entropy(p_n) + 1e-12)

    p_n_safe = np.clip(p_n, 1e-12, None)
    p_1_m = np.clip(p_1_m, 1e-12, None)
    p_2_m = np.clip(p_2_m, 1e-12, None)
    d1 = np.sum(p_n_safe * np.log2(p_n_safe / p_1_m))
    d2 = np.sum(p_n_safe * np.log2(p_n_safe / p_2_m))
    acc2 = (d1 - d2) / (d1 + 1e-12)

    return acc1, acc2


def ener_calculate_pca_bin(binarizedData):
    h, J = fit_exact_mem(
        binarizedData,
        max_iter=20000,
        alpha=0.2,
        tol=1e-5,
        print_every=200,
        anneal=0.985,
        clip=0.05
    )
    acc_1, acc_2 = calc_accuracy(h, J, binarizedData)
    energy_one = calc_energy(h, J, binarizedData)
    return h, J, energy_one
