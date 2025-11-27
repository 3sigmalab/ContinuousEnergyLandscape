import os, sys

sys.path.append(os.path.abspath(".."))

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from continuous_energy_landscape import ContinuousEnergyLandscape
from simulations.slds_sim_simple import simulate_slds_example
from metrics_energy_landscape import compute_metrics_all
from discrete_energy import binarize, ener_calculate_pca_bin

K = 3          # number of discrete states
d = 8          # dimensionality
T = 4000       # number of time points
seed = 0

sim, X, g_true = simulate_slds_example(
    K=K,
    d=d,
    T=T,
    scenario="nonlinear_obs",  # <--- key change
    dwell=100,                 # slightly longer dwell
    sigma=0.05,                # add some observation noise
    seed=seed,
)

P_true = sim.P
print("X shape:", X.shape)
print("g_true shape:", g_true.shape)


# X has shape (T, d); discrete_energy expects (D, T)
X_bin = binarize(X.T)
h_del, W_del, E_del = ener_calculate_pca_bin(X_bin)

print("Discrete energy shape:", E_del.shape)

# k-means on one-dimensional energy feature
E_feat = E_del.reshape(-1, 1)
km_del = KMeans(n_clusters=K, random_state=seed)
g_del = km_del.fit_predict(E_feat)
print("DEL labels shape:", g_del.shape)


cel = ContinuousEnergyLandscape(
    hidden_channels=64,
    rank=32,
    delta=0.10,
    eps=1e-2,
    lambda_reg=1e-2,
    lr=1e-3,
    weight_decay=0.0,
    max_epochs=300,
    clip_grad=1.0,
    verbose=True,
    device="cpu",
    seed=seed,
)

out = cel.fit(X)
E_cel = cel.predict_energy(X)
print("CEL energy shape:", E_cel.shape)

# 1D embedding from top eigenvector of S
S = cel.S_
eigvals, eigvecs = np.linalg.eigh(S)
v_top = eigvecs[:, np.argmax(eigvals)]
z_cel = (X @ v_top).reshape(-1, 1)

F_cel = np.hstack([E_cel.reshape(-1, 1), z_cel])
km_cel = KMeans(n_clusters=K, random_state=seed)
g_cel = km_cel.fit_predict(F_cel)
print("CEL labels shape:", g_cel.shape)


metrics_del = compute_metrics_all(g_true, g_del, K=K, P_true=P_true, X=X)
metrics_cel = compute_metrics_all(g_true, g_cel, K=K, P_true=P_true, X=X)

print("DEL metrics:", metrics_del)
print("CEL metrics:", metrics_cel)

print("DEL ARI:", adjusted_rand_score(g_true, g_del))
print("CEL ARI:", adjusted_rand_score(g_true, g_cel))