import os, sys

sys.path.append(os.path.abspath(".."))


import numpy as np
from continuous_energy_landscape import ContinuousEnergyLandscape

# --- synthetic Gaussian data ---
T, D = 500, 20
rng = np.random.RandomState(0)

# simple correlated covariance matrix
Sigma = 0.5 * np.eye(D)
for i in range(D):
    for j in range(i + 1, D):
        Sigma[i, j] = Sigma[j, i] = 0.3 * np.exp(-abs(i - j) / 5.0)

X = rng.multivariate_normal(mean=np.zeros(D), cov=Sigma, size=T).astype(np.float32)

cel = ContinuousEnergyLandscape(
    hidden_channels=64,
    rank=16,
    delta=0.10,
    eps=1e-2,
    lambda_reg=1e-2,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=500,
    clip_grad=1.0,
    verbose=True,
    device="cpu",
    seed=0,
)

out = cel.fit(X)
print("Best training loss:", out["best_loss"])
print("Covariance fit metrics:", out["metrics"])

E = cel.predict_energy(X)
print("Energy shape:", E.shape)
print("Energy range: min=", E.min(), " max=", E.max())