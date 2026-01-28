# methods.py
import numpy as np
from scipy.special import gammaln


# =============================================================================
# 1) Math utilities (Vectorized)
# =============================================================================
def log_comb_vectorized(n, k):
    """Computes log(C(n, k)) robustly."""
    n = np.asanyarray(n)
    k = np.asanyarray(k)
    out_shape = np.broadcast(n, k).shape
    res = np.full(out_shape, -np.inf, dtype=np.float64)
    mask = (k >= 0) & (k <= n)
    if np.any(mask):
        n_safe = np.where(mask, n, 1.0)
        k_safe = np.where(mask, k, 1.0)
        val = gammaln(n_safe + 1) - gammaln(k_safe + 1) - gammaln(n_safe - k_safe + 1)
        np.place(res, mask, val[mask])
    return res


def get_nhg_pmf_vectorized(n_calib, m_test, r_c_array):
    """Computes PMF of Negative Hypergeometric Distribution."""
    k_vals = np.arange(m_test + 1)
    N = n_calib + m_test
    log_denom = log_comb_vectorized(N, m_test)
    r_c_col = r_c_array[:, None]
    k_row = k_vals[None, :]
    term1 = log_comb_vectorized(r_c_col + k_row - 1, k_row)
    term2 = log_comb_vectorized(N - r_c_col - k_row, m_test - k_row)
    log_pmf = term1 + term2 - log_denom
    pmf = np.exp(log_pmf)
    row_sums = np.sum(pmf, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pmf /= row_sums
    return pmf


def sample_nhg_vectorized(n_calib, m_test, r_c_array, size=1, rng=None):
    """Samples from NHG distribution."""
    if rng is None: rng = np.random.default_rng()
    pmf = get_nhg_pmf_vectorized(n_calib, m_test, r_c_array)
    cdf = np.cumsum(pmf, axis=1)
    n_items = len(r_c_array)
    rand_vals = rng.random((n_items, size))
    samples = (cdf[:, None, :] < rand_vals[:, :, None]).sum(axis=2)
    return samples.squeeze() if size == 1 else samples


# =============================================================================
# 2) Helpers
# =============================================================================
def rank_1_to_n(values):
    """Converts scores/values to 1-based ranks."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def eval_fcp_and_rl(pred_sets, true_ranks, total_size):
    """Evaluates Coverage (FCP), Relative Length, and Avg Set Size."""
    true_ranks = np.asarray(true_ranks, dtype=int)
    cover = 0
    lengths = []
    for ps, r in zip(pred_sets, true_ranks):
        if ps and len(ps) == 2:
            l, h = ps
            if l <= r <= h:
                cover += 1
            length = h - l + 1
            lengths.append(length)
        else:
            lengths.append(0)

    fcp = cover / len(true_ranks)
    avg_len = float(np.mean(lengths))
    rel_len = avg_len / float(total_size)
    return fcp, rel_len, avg_len


def cp_threshold(scores, alpha):
    """Standard Conformal Prediction Quantile"""
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    k = np.clip(k, 0, n - 1)
    return float(s[k])


# =============================================================================
# 3) Prediction Sets
# =============================================================================
def prediction_sets_from_threshold(pred_test, threshold, mode, total_size, V_sorted=None):
    pred_test = np.asarray(pred_test)
    pred_sets = []

    if mode == "ra":
        q_int = int(np.ceil(threshold))
        for r_hat in pred_test.astype(int):
            l = max(1, int(r_hat - q_int))
            h = min(total_size, int(r_hat + q_int))
            pred_sets.append([l, h])

    elif mode == "va":
        if V_sorted is None: raise ValueError("V_sorted required for VA")
        V_ref = np.asarray(V_sorted)
        low_vals = pred_test - threshold
        high_vals = pred_test + threshold

        idx_l = np.searchsorted(V_ref, low_vals, side="left")
        idx_r = np.searchsorted(V_ref, high_vals, side="right")

        l_ranks = np.maximum(1, idx_l + 1)
        h_ranks = np.minimum(total_size, idx_r)

        for l, h in zip(l_ranks, h_ranks):
            if l <= h:
                pred_sets.append([int(l), int(h)])
            else:
                pred_sets.append([])
    return pred_sets


# =============================================================================
# 4) DCR & MDCR Thresholds
# =============================================================================
def dcr_threshold_exact(n_calib, m_test, Rc, pred_calib, alpha, mode, V_sorted=None):
    Rc = np.asarray(Rc, dtype=int)
    pmf_k = get_nhg_pmf_vectorized(n_calib, m_test, Rc)
    k_vals = np.arange(m_test + 1)[None, :]
    R_true = Rc[:, None] + k_vals
    target_prob = np.ceil((n_calib + 1) * (1 - alpha)) / (n_calib + 1)

    if mode == "ra":
        R_hat = pred_calib.astype(int)[:, None]
        S = np.abs(R_true - R_hat).astype(int)
        flat_S = S.ravel()
        flat_P = (pmf_k / n_calib).ravel()
        pdf = np.bincount(flat_S, weights=flat_P, minlength=int(n_calib + m_test + 2))
        cdf = np.cumsum(pdf)
        return float(np.argmax(cdf >= target_prob))
    elif mode == "va":
        idx = np.clip(R_true - 1, 0, len(V_sorted) - 1)
        V_true = V_sorted[idx]
        V_hat = pred_calib[:, None]
        S = np.abs(V_hat - V_true).ravel()
        P = (pmf_k / n_calib).ravel()
        ord_ = np.argsort(S)
        cum = np.cumsum(P[ord_])
        return float(S[ord_][int(np.argmax(cum >= target_prob))])


def mdcr_threshold_sampling(n_calib, m_test, Rc, pred_calib, alpha, mode, V_sorted=None, B=100, rng=None):
    K_samp = sample_nhg_vectorized(n_calib, m_test, Rc, size=B, rng=rng)
    R_samp = Rc[:, None] + K_samp
    target_prob = np.ceil((n_calib + 1) * (1 - alpha)) / (n_calib + 1)

    if mode == "ra":
        R_hat = pred_calib.astype(int)[:, None]
        S = np.abs(R_samp - R_hat).ravel().astype(float)
    else:
        idx = np.clip(R_samp - 1, 0, len(V_sorted) - 1)
        V_true = V_sorted[idx]
        V_hat = pred_calib[:, None]
        S = np.abs(V_hat - V_true).ravel().astype(float)

    S_sorted = np.sort(S)
    j = int(np.ceil(target_prob * len(S_sorted))) - 1
    return float(S_sorted[np.clip(j, 0, len(S_sorted) - 1)])


# =============================================================================
# 5) Oracle & TCPR Utilities
# =============================================================================
def oracle_threshold(V_sorted, r_true_full, calib_idx, pred_calib, mode, alpha):
    """Oracle uses True Absolute Ranks"""
    r_true_calib = r_true_full[calib_idx]
    if mode == "ra":
        s = np.abs(r_true_calib - pred_calib.astype(int))
    else:
        v_ref = V_sorted[r_true_calib - 1]
        s = np.abs(pred_calib - v_ref)
    return cp_threshold(s, alpha)


def hhp_simulate_quantile_envelopes(n, m, K=1000, delta=0.02, rng=None):
    """Simulates Quantile Envelopes for TCPR (Assumption of Uniformity)"""
    if rng is None: rng = np.random.default_rng(2026)
    total = n + m
    # Vectorized simulation of ranks
    # Simulate K trials of n calibration items among n+m total
    all_ranks = np.zeros((n, K), dtype=int)

    # Efficient simulation: sample n indices from N, sort them.
    # We do this K times.
    for k in range(K):
        # Sampling n positions from 1..N without replacement
        ranks = np.sort(rng.choice(total, size=n, replace=False) + 1)
        all_ranks[:, k] = ranks

    gamma = delta / 2.0
    R_minus = np.quantile(all_ranks, gamma, axis=1).astype(int)
    R_plus = np.quantile(all_ranks, 1.0 - gamma, axis=1).astype(int)
    return R_minus, R_plus


def tcpr_proxy_scores(Rc, pred_calib, mode, V_sorted, R_minus, R_plus, total_size):
    """Calculates proxy scores for TCPR based on envelopes"""
    # Rc is relative rank (1..n), used to index the envelope arrays
    pos = np.asarray(Rc, dtype=int) - 1

    # Get bounds for each item i based on its relative rank
    rmin = np.clip(R_minus[pos], 1, total_size)
    rmax = np.clip(R_plus[pos], 1, total_size)

    if mode == "ra":
        R_hat = pred_calib.astype(int)
        # Worst-case distance to the interval [rmin, rmax]
        # Actually TCPR typically takes max( |R_hat - rmin|, |R_hat - rmax| )
        return np.maximum(np.abs(R_hat - rmin), np.abs(R_hat - rmax)).astype(float)
    else:
        vmin = V_sorted[rmin - 1]
        vmax = V_sorted[rmax - 1]
        V_hat = np.asarray(pred_calib)
        return np.maximum(np.abs(V_hat - vmin), np.abs(V_hat - vmax)).astype(float)