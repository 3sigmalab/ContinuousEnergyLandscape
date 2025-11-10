
"""Minimal Kuramoto-based simulator + helper for CEL vs DEL examples.

This is a trimmed version of your earlier Kuramoto experiment script.
It exposes:

- KuramotoBalloonSim : K-state Kuramoto + hemodynamic model
- simulate_kuramoto_example : helper that returns (sim, X_bold, g_true)
"""
import numpy as np
from scipy.signal import hilbert

class KuramotoBalloonSim:
    """Hidden state k controls coupling gain in Kuramoto dynamics.

    Neural activity (sin(theta)) is convolved with a simple hemodynamic kernel
    to yield BOLD-like signals.
    """ 
    def __init__(self, K: int, d: int, seed: int = 0, dt: float = 0.5,
                 dwell: int = 120, k_low: float = 0.05, k_high: float = 0.35):
        rng = np.random.default_rng(seed)
        self.K, self.d, self.dt = K, d, dt
        self.k_low, self.k_high = k_low, k_high

        # Symmetric coupling (zero diag), normalized
        W = rng.random((d, d))
        W = (W + W.T) / 2.0
        np.fill_diagonal(W, 0.0)
        W /= (W.max() + 1e-6)
        self.W = W

        # Intrinsic frequencies (small spread)
        self.natural = rng.normal(0.0, 0.05, size=d)

        # HMM transitions (mean dwell)
        p_stay = 1.0 - 1.0 / float(dwell)
        P = np.full((K, K), (1 - p_stay) / (K - 1))
        np.fill_diagonal(P, p_stay)
        self.P = P

        # Simple hemodynamic kernel (proxy)
        h = np.exp(-np.arange(40) / 6.0)
        self.h = h / h.sum()

    def _k_for_state(self, k: int) -> float:
        if self.K == 1:
            return self.k_high
        return self.k_low + (self.k_high - self.k_low) * (k / (self.K - 1))

    def sample(self, T: int, eta: float = 0.0, sigma: float = 0.6, seed: int = 0):
        """Generate a trajectory of BOLD-like signals and hidden regimes.

        Returns
        -------
        X_bold : ndarray, shape (T, d)
            BOLD-like signals.
        g : ndarray, shape (T,)
            Hidden regime indices.
        TH : ndarray, shape (T, d)
            Phases in radians.
        """ 
        rng = np.random.default_rng(seed)

        # Hidden states
        g = np.empty(T, dtype=int)
        g[0] = rng.integers(self.K)
        for t in range(1, T):
            g[t] = rng.choice(self.K, p=self.P[g[t - 1]])

        # Kuramoto phases
        theta = rng.uniform(-np.pi, np.pi, size=self.d)
        TH = np.zeros((T, self.d))
        neural = np.zeros((T, self.d))
        noise_scale = 0.02 * sigma

        for t in range(T):
            k_t = self._k_for_state(g[t])
            sin_term = (self.W * np.sin(theta[:, None] - theta[None, :])).sum(axis=1)
            dtheta = self.natural + k_t * sin_term + noise_scale * rng.normal(size=self.d)
            theta = (theta + self.dt * dtheta) % (2 * np.pi)
            TH[t] = theta
            neural[t] = np.sin(theta)

        # Hemodynamic smoothing per ROI
        X = np.vstack(
            [np.convolve(neural[:, j], self.h, mode="same") for j in range(self.d)]
        ).T
        if sigma > 0.0:
            X = X + rng.normal(scale=0.05 * sigma, size=X.shape)
        return X, g, TH

def _stationary_from_P(P: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, idx])
    pi = np.maximum(pi, 0)
    s = pi.sum()
    return (pi / s) if s > 0 else np.ones(P.shape[0]) / P.shape[0]

def simulate_kuramoto_example(K: int = 3, d: int = 30, T: int = 4000,
                              dwell: int = 140, sigma: float = 0.6,
                              k_low: float = 0.06, k_high: float = 0.34,
                              seed: int = 0):
    """Convenience wrapper to simulate one Kuramoto-BOLD trajectory.

    Returns
    -------
    sim : KuramotoBalloonSim
        The simulator object (contains true P).
    X_bold : ndarray, shape (T, d)
        BOLD-like signals.
    g_true : ndarray, shape (T,)
        Ground-truth hidden regimes.
    """ 
    sim = KuramotoBalloonSim(K=K, d=d, seed=seed, dwell=dwell,
                             k_low=k_low, k_high=k_high)
    X, g_true, TH = sim.sample(T=T, sigma=sigma, seed=seed + 2)
    return sim, X, g_true
