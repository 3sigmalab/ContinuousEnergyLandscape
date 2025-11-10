
"""Minimal SLDS simulator + helper for CEL vs DEL examples.

This is a cleaned-up version of your earlier SLDS experiment script, intended
for easy reuse and example notebooks. It exposes:

- SwitchingVARSim : K-state VAR(1) simulator
- simulate_slds_example : helper that returns (sim, X, g_true)
"""
import numpy as np

class SwitchingVARSim:
    """Switching VAR(1) model used as an SLDS-like simulator.

    Dynamics:
        x_t = mu_k + A_k (x_{t-1} - mu_k) + eps_k,  where k is the discrete state.

    The scenario argument follows your original code:
    - 'default'    : different means, generic covariances
    - 'var_only'   : same means, different dynamics / noise scales
    - 'cov_orient' : same means, different covariance orientations
    - 'nonlinear_obs' : latent VAR(1) passed through tanh + small noise
    """ 
    def __init__(self, K: int, d: int, dwell: int = 80, seed: int = 0,
                 scenario: str = "default", nonlinear_obs: bool = False):
        rng = np.random.default_rng(seed)
        self.K, self.d = K, d
        self.scenario = scenario
        self.nonlinear_obs = nonlinear_obs

        # KxK transition with mean dwell
        p_stay = 1.0 - 1.0 / float(dwell)
        P = np.full((K, K), (1 - p_stay) / (K - 1))
        np.fill_diagonal(P, p_stay)
        self.P = P
        self.pi = np.ones(K) / K

        # State parameters: x_t = mu_k + A_k (x_{t-1} - mu_k) + eps_k
        if scenario == "var_only":
            self.mus = np.zeros((K, d))
            A_base = rng.normal(scale=0.10, size=(d, d))
            # stabilize
            u, s, vt = np.linalg.svd(A_base)
            A_base = (A_base / (s.max() + 1e-6)) * 0.7
            self.Aks, self.Sigmas = [], []
            scales = np.linspace(0.6, 1.8, K)
            for k in range(K):
                A = A_base + rng.normal(scale=0.02, size=(d, d))
                self.Aks.append(A)
                B = rng.normal(size=(d, d))
                Sigma = (B @ B.T) / (d ** 2)
                tr = np.trace(Sigma) / d
                Sigma = Sigma / (tr + 1e-9)
                Sigma *= (scales[k] ** 2) * 0.05
                Sigma += 0.005 * np.eye(d)
                self.Sigmas.append(Sigma)

        elif scenario == "cov_orient":
            self.mus = np.zeros((K, d))
            self.Aks, self.Sigmas = [], []
            A_base = rng.normal(scale=0.10, size=(d, d))
            u, s, vt = np.linalg.svd(A_base)
            A_base = (A_base / (s.max() + 1e-6)) * 0.7
            eigvals = np.linspace(0.03, 0.10, d)  # matched marginal variance, different orientations
            for k in range(K):
                A = A_base + rng.normal(scale=0.02, size=(d, d))
                self.Aks.append(A)
                Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
                Sigma = (Q * eigvals) @ Q.T
                Sigma += 0.003 * np.eye(d)
                self.Sigmas.append(Sigma)

        elif scenario == "nonlinear_obs":
            self.mus = np.zeros((K, d))
            A_base = rng.normal(scale=0.10, size=(d, d))
            u, s, vt = np.linalg.svd(A_base)
            A_base = (A_base / (s.max() + 1e-6)) * 0.7
            self.Aks, self.Sigmas = [], []
            scales = np.linspace(0.6, 1.8, K)
            for k in range(K):
                A = A_base + rng.normal(scale=0.02, size=(d, d))
                self.Aks.append(A)
                B = rng.normal(size=(d, d))
                Sigma = (B @ B.T) / (d ** 2)
                tr = np.trace(Sigma) / d
                Sigma = Sigma / (tr + 1e-9)
                Sigma *= (scales[k] ** 2) * 0.05
                Sigma += 0.005 * np.eye(d)
                self.Sigmas.append(Sigma)
            self.nonlinear_obs = True

        else:  # 'default': means differ, generic covariances
            self.mus = rng.normal(scale=0.3, size=(K, d))
            self.Aks, self.Sigmas = [], []
            for _ in range(K):
                A = rng.normal(scale=0.15, size=(d, d))
                u, s, vt = np.linalg.svd(A)
                A = (A / (s.max() + 1e-6)) * 0.8
                self.Aks.append(A)
                B = rng.normal(size=(d, d))
                Sigma = (B @ B.T) / (d ** 2) + 0.02 * np.eye(d)
                self.Sigmas.append(Sigma)

    def sample(self, T: int, eta: float = 0.0, sigma: float = 0.0, seed: int = 0):
        """Generate a trajectory of length T.

        Returns
        -------
        X : ndarray, shape (T, d)
            Observations.
        g : ndarray, shape (T,)
            Discrete regime indices in {0, ..., K-1}.
        """ 
        rng = np.random.default_rng(seed)
        g = np.empty(T, dtype=int)
        X = np.empty((T, self.d))
        g[0] = rng.integers(self.K)
        k = g[0]
        X[0] = self.mus[k] + rng.multivariate_normal(np.zeros(self.d), self.Sigmas[k])
        for t in range(1, T):
            g[t] = rng.choice(self.K, p=self.P[g[t - 1]])
            k = g[t]
            X[t] = (
                self.mus[k]
                + self.Aks[k] @ (X[t - 1] - self.mus[k])
                + rng.multivariate_normal(np.zeros(self.d), self.Sigmas[k])
            )
        if self.nonlinear_obs:
            X = np.tanh(X) + rng.normal(scale=0.02, size=X.shape)
        if sigma > 0.0:
            X = X + rng.normal(scale=sigma, size=X.shape)
        return X, g

def simulate_slds_example(K: int = 3, d: int = 8, T: int = 5000,
                          scenario: str = "default", dwell: int = 80,
                          sigma: float = 0.0, seed: int = 0):
    """Convenience wrapper to simulate one SLDS trajectory.

    Returns
    -------
    sim : SwitchingVARSim
        The simulator object (contains true P and mus).
    X : ndarray, shape (T, d)
        Time series.
    g_true : ndarray, shape (T,)
        Ground-truth discrete states.
    """ 
    sim = SwitchingVARSim(K=K, d=d, dwell=dwell, seed=seed, scenario=scenario)
    X, g_true = sim.sample(T=T, sigma=sigma, seed=seed + 1)
    return sim, X, g_true
