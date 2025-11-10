
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from typing import Tuple, Dict, Any, Optional

# ---------- Utilities ----------

def zscore_np(X: np.ndarray, axis: int = 0, eps: float = 1e-8) -> np.ndarray:
    mean = X.mean(axis=axis, keepdims=True)
    std = X.std(axis=axis, keepdims=True)
    return (X - mean) / (std + eps)

def build_graph_from_corr(
    X_z, delta=0.10, device=None, fisher=True
):
    D = X_z.shape[1]
    R = np.corrcoef(X_z, rowvar=False)
    np.fill_diagonal(R, 0.0)
    i, j = np.triu_indices(D, 1)
    tau = np.quantile(np.abs(R[i, j]), 1.0 - delta)      # keep top-|corr|
    mask = (np.abs(R) >= tau)
    np.fill_diagonal(mask, False)

    src, dst = np.where(mask)
    w = R[src, dst]                                      # *** keep SIGN ***
    if fisher:
        w = np.arctanh(np.clip(w, -0.999, 0.999))        # Fisher z

    # undirected
    src_full = np.hstack([src, dst]); dst_full = np.hstack([dst, src])
    w_full = np.hstack([w, w])

    edge_index = torch.tensor(np.vstack([src_full, dst_full]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(np.asarray(w_full, dtype=np.float32), dtype=torch.float32, device=device)
    return edge_index, edge_weight, R

def build_signed_graph_from_corr(X_z, delta=0.10, device=None, fisher=True):
    D = X_z.shape[1]
    R = np.corrcoef(X_z, rowvar=False); np.fill_diagonal(R, 0.0)
    iu, ju = np.triu_indices(D, 1)
    tau = np.quantile(np.abs(R[iu, ju]), 1.0 - delta)

    pos = (R >= tau); neg = (R <= -tau)
    np.fill_diagonal(pos, False); np.fill_diagonal(neg, False)

    def to_edge(Rmask, vals):
        si, sj = np.where(Rmask)
        w = vals[si, sj]
        if fisher:
            w = np.arctanh(np.clip(w, -0.999, 0.999))
        w = np.abs(w)  # make weights non-negative for GCN normalization
        # undirected duplication
        src = np.hstack([si, sj]); dst = np.hstack([sj, si]); w = np.hstack([w, w])
        ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long, device=device)
        ew = torch.tensor(np.asarray(w, dtype=np.float32), dtype=torch.float32, device=device)
        return ei, ew

    edge_pos, w_pos = to_edge(pos, R)
    edge_neg, w_neg = to_edge(neg, -R)  # magnitude of negatives
    return edge_pos, w_pos, edge_neg, w_neg, R


def cholesky_nll_with_reg(
    X: torch.Tensor,   # (T, D)
    S: torch.Tensor,   # (D, D) precision (PD)
    h: torch.Tensor,   # (D,)
    lambda_reg: float = 1e-2
) -> torch.Tensor:
    """
    Correct Gaussian NLL for precision S:
      NLL = 0.5 * sum_t (x_t - mu)^T S (x_t - mu) - 0.5 * T * logdet(S) + lambda ||S||_F^2
    where mu = S^{-1} h.
    """
    T, D = X.shape
    mu = torch.linalg.solve(S, h.unsqueeze(-1)).squeeze(-1)   # (D,)

    diff = X - mu                                             # (T, D)
    L = torch.linalg.cholesky(S)                              # S = L L^T

    # Quadratic term with S (NOT S^{-1}):
    # (x - mu)^T S (x - mu) = || L^T (x - mu) ||^2 = || (x - mu) @ L ||^2
    v = diff @ L                                              # (T, D)
    term_quadratic = 0.5 * (v * v).sum()

    logdet = 2.0 * torch.log(torch.diagonal(L)).sum()
    nll = term_quadratic - 0.5 * T * logdet
    return nll + lambda_reg * (S * S).sum()

def compute_energies_np(X: np.ndarray, S: np.ndarray, mu: np.ndarray) -> np.ndarray:
    diff = X - mu
    return 0.5 * np.einsum("ti,ij,tj->t", diff, S, diff, optimize=True)

def evaluate_covariance_fit_np(W: np.ndarray, h: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
    D = X.shape[1]
    emp_mu = X.mean(axis=0)
    Xc = X - emp_mu
    emp_cov = np.cov(Xc, rowvar=False)
    S_est = -W
    try:
        Sigma_est = np.linalg.inv(S_est)
    except np.linalg.LinAlgError:
        return dict(cov_fro_err=np.nan, cov_fro_rel=np.nan, cov_r=np.nan, mean_err=np.nan)
    mu_est = np.linalg.solve(S_est, h)
    fro_err = np.linalg.norm(Sigma_est - emp_cov, "fro")
    fro_rel = fro_err / (np.linalg.norm(emp_cov, "fro") + 1e-12)
    i_up, j_up = np.triu_indices(D, k=1)
    model_vals = Sigma_est[i_up, j_up]
    emp_vals = emp_cov[i_up, j_up]
    cov_r = np.corrcoef(model_vals, emp_vals)[0, 1]
    mean_err = np.linalg.norm(mu_est - emp_mu)
    return dict(cov_fro_err=fro_err, cov_fro_rel=fro_rel, cov_r=cov_r, mean_err=mean_err)

# ---------- Model ----------

class ReparamEnergyLandscapeGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_nodes, rank=96, eps=1e-2):
        super().__init__()
        self.num_nodes, self.rank, self.eps = num_nodes, rank, eps
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.LayerNorm(hidden_channels),
        )
        self.gcn_pos = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.gcn_neg = GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True)
        self.alpha_pos = nn.Parameter(torch.tensor(0.5))
        self.alpha_neg = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(0.1)

        self.factor = nn.Linear(hidden_channels, rank, bias=False)
        self.free_U = nn.Parameter(torch.zeros(num_nodes, rank))
        self.h_head = nn.Linear(hidden_channels, 1)

        nn.init.kaiming_uniform_(self.factor.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.h_head.weight, a=np.sqrt(5))
        nn.init.zeros_(self.h_head.bias)

    def forward(self, x, edge_pos, w_pos, edge_neg, w_neg):
        H0  = self.mlp(x)
        Hgp = self.gcn_pos(H0, edge_pos, edge_weight=w_pos)
        Hgn = self.gcn_neg(H0, edge_neg, edge_weight=w_neg)
        H   = H0 + self.alpha_pos*Hgp - self.alpha_neg*Hgn
        H   = self.dropout(H)

        Z = self.factor(H) + self.free_U
        S = Z @ Z.t() + self.eps * torch.eye(self.num_nodes, device=Z.device)  # PD
        W = -S
        h = self.h_head(H).squeeze(-1)
        return W, h



# ---------- End-to-end wrapper ----------

class ContinuousEnergyLandscape:
    def __init__(
        self,
        hidden_channels: int = 64,
        rank: int = 32,
        delta: float = 0.10,
        eps: float = 1e-4,
        lambda_reg: float = 1e-2,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 2000,
        clip_grad: float = 1.0,
        verbose: bool = True,
        device: str = "cpu",
        seed: int = 0,
    ):
        self.hidden_channels = hidden_channels
        self.rank = rank
        self.delta = delta
        self.eps = eps
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.clip_grad = clip_grad
        self.verbose = verbose
        self.device = torch.device(device)
        self.seed = seed

        self.model = None
        self.edge_index = None
        self.edge_weight = None
        self.W_ = None
        self.h_ = None
        self.S_ = None
        self.mu_ = None
        self.corr_ = None
        self.mean_ = None
        self.std_ = None
        self.metrics_ = None

    def _set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

    def fit(self, X: np.ndarray) -> Dict[str, Any]:
        self._set_seed()
        X = np.asarray(X, dtype=np.float32)
        T, D = X.shape
        # standardize using training mean/std and store for reuse
        self.mean_ = X.mean(axis=0, keepdims=True)
        self.std_ = X.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-8] = 1.0
        Xz = (X - self.mean_) / self.std_

        # --- build signed graphs ---
        edge_pos, w_pos, edge_neg, w_neg, R = build_signed_graph_from_corr(
            Xz, delta=self.delta, device=self.device, fisher=True
        )
        self.edge_pos, self.w_pos = edge_pos, w_pos
        self.edge_neg, self.w_neg = edge_neg, w_neg
        self.corr_ = R

        # node features (per node: its T-length timeseries)
        X_node = torch.tensor(Xz.T, dtype=torch.float32, device=self.device)  # (D, T)
        X_seq  = torch.tensor(Xz,   dtype=torch.float32, device=self.device)  # (T, D)

        rank_eff = min(self.rank, D)
        self.model = ReparamEnergyLandscapeGNN(
            in_channels=T,
            hidden_channels=self.hidden_channels,
            num_nodes=D,
            rank=rank_eff,
            eps=self.eps,          # recommend eps=1e-2 for robustness
        ).to(self.device)

        # optimizer (recommend no weight decay here)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float("inf")
        best_state = None
        self.model.train()

        for epoch in range(self.max_epochs):
            opt.zero_grad()
            # single forward on signed graphs
            W_pred, h_pred = self.model(X_node, self.edge_pos, self.w_pos, self.edge_neg, self.w_neg)
            S_pred = -W_pred
            loss = cholesky_nll_with_reg(X_seq, S_pred, h_pred, lambda_reg=self.lambda_reg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            opt.step()

            if self.verbose and (epoch % 100 == 0 or epoch == self.max_epochs - 1):
                print(f"[epoch {epoch:04d}] loss={loss.item():.6f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        with torch.no_grad():
            # final forward uses the SAME signed graphs
            W_pred, h_pred = self.model(X_node, self.edge_pos, self.w_pos, self.edge_neg, self.w_neg)
            W_np = np.asarray(W_pred.detach().cpu().tolist(), dtype=np.float32)
            h_np = np.asarray(h_pred.detach().cpu().tolist(), dtype=np.float32)
            S_np = -W_np
            mu_np = np.linalg.solve(S_np, h_np)

        self.W_, self.h_, self.S_, self.mu_ = W_np, h_np, S_np, mu_np
        self.metrics_ = evaluate_covariance_fit_np(self.W_, self.h_, Xz)

        return dict(
            W=self.W_, h=self.h_, S=self.S_, mu=self.mu_,
            metrics=self.metrics_, corr=self.corr_, best_loss=best_loss
        )


    def predict_energy(self, X: np.ndarray) -> np.ndarray:
        assert self.S_ is not None and self.mu_ is not None, "Call fit() first."
        X = np.asarray(X, dtype=np.float32)
        assert self.mean_ is not None and self.std_ is not None, "Model has no stored normalization stats; call fit() first."
        Xz = (X - self.mean_) / self.std_
        return compute_energies_np(Xz, self.S_, self.mu_)

    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.W_, self.h_, self.S_, self.mu_

    def evaluate(self, X: np.ndarray) -> Dict[str, Any]:
        X = np.asarray(X, dtype=np.float32)
        assert self.mean_ is not None and self.std_ is not None, "Model has no stored normalization stats; call fit() first."
        Xz = (X - self.mean_) / self.std_
        return evaluate_covariance_fit_np(self.W_, self.h_, Xz)
