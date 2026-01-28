# main.py
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import model as mdl  # Import model.py
import methods as mtd  # Import methods.py

# =============================================================================
# Global Config
# =============================================================================
N_CALIB_LIST = [500]  # Calibration set sizes to test
M_TEST_LIST = [1000]  # Test set sizes to test
B_MDCR = 100  # Monte Carlo samples for MDCR
ALPHA = 0.1  # Target miscoverage level
D = 20  # Feature dimension
SIGMA = 0.2  # Noise level
N_TR = 1000  # Training set size
N_SIM = 100  # Number of trials per config

# Output Paths
OUT_DIR = os.path.join("results")
os.makedirs(OUT_DIR, exist_ok=True)
CSV_PATH = os.path.join(OUT_DIR, "synthetic_results_final.csv")

# =============================================================================
# Synthetic Data Generation (Consistent Ground Truth)
# =============================================================================
_GLOBAL_W = None


def reset_global_model():
    """Resets global model parameters to ensure a fresh experiment starts cleanly."""
    global _GLOBAL_W
    _GLOBAL_W = None


def generate_synthetic_data(n_total, d=D, sigma=SIGMA, seed=None):
    """
    Generates synthetic regression data: y = w^T x + noise.

    CRITICAL: This function uses a persistent global weight vector (_GLOBAL_W).
    This ensures that the Training Set and the Test/Calibration Pool are generated
    from the EXACT SAME underlying distribution (Ground Truth), which is essential
    for valid statistical calibration experiments.
    """
    global _GLOBAL_W
    if _GLOBAL_W is None:
        # Initialize true weights only once
        rng_w = np.random.default_rng(42)
        _GLOBAL_W = rng_w.normal(0, 1, d)
        _GLOBAL_W /= np.linalg.norm(_GLOBAL_W)
        print(f"[DataGen] Initialized Global Ground Truth W (d={d}).")

    rng = np.random.default_rng(seed)

    # Generate features X ~ N(0, 1)
    X = rng.normal(0, 1, (n_total, d)).astype(np.float32)

    # Calculate True Logits
    logits = np.dot(X, _GLOBAL_W)

    # Add Gaussian Noise
    noise = rng.normal(0, sigma, n_total)
    Y = logits + noise

    return X, Y.astype(np.float32)


# Wrappers for semantic clarity
def generate_training_data(n_tr=N_TR, d=D, sigma=SIGMA, seed=None):
    return generate_synthetic_data(n_tr, d=d, sigma=sigma, seed=seed)


def generate_finite_population_pool(M_total, d=D, sigma=SIGMA, seed=None):
    return generate_synthetic_data(M_total, d=d, sigma=sigma, seed=seed)


# =============================================================================
# Main Execution Flow
# =============================================================================
def main():
    print(">>> Main Execution Started: Synthetic Linear Experiment")

    # 1. Initialize Ground Truth
    reset_global_model()

    # 2. Generate Training Data
    print(f">>> Generating Training Data (N={N_TR}, D={D}, Sigma={SIGMA})...")
    X_tr, Y_tr = generate_training_data(N_TR, d=D, sigma=SIGMA, seed=42)

    # 3. Train Base Model (RankNet)
    print("\n>>> Training RankNet Base Model...")
    # model_obj is a tuple: (pytorch_model, scaler)
    model_obj = mdl.train_ranknet(X_tr, Y_tr, epochs=200, batch_size=128)

    # 4. Prepare Results Storage
    columns = ["Dataset", "Model", "Score_Type", "n", "m", "Algorithm", "Trial", "FCP", "Relative_Length", "Set_Size"]
    all_results = []

    # 5. Experiment Loop (Varying n and m)
    for n in N_CALIB_LIST:
        for m in M_TEST_LIST:
            N_TOTAL = n + m
            print(f"\n  Running Configuration: n={n}, m={m}")

            # Generate a fresh finite population pool for this trial configuration
            # Note: It uses the SAME _GLOBAL_W, so it's from the same distribution as Training Data.
            X_pool, Y_pool = generate_finite_population_pool(N_TOTAL, d=D, sigma=SIGMA, seed=None)

            # Predict on full pool using the trained RankNet
            scores_full, _ = mdl.predict_ranknet(model_obj, X_pool)
            scores_full = np.asarray(scores_full).reshape(-1)

            # Reference System for VA (Sorted Scores of the population)
            V_sorted = np.sort(scores_full)

            # Predicted Ranks (R_hat)
            r_hat_full = mtd.rank_1_to_n(scores_full)

            # True Ranks (R_true) - Needed for evaluation ground truth
            r_true_full = mtd.rank_1_to_n(Y_pool)

            # Run Independent Trials (Repeated Random Subsampling)
            for sim in tqdm(range(N_SIM), desc="    > Trials"):
                # Random split into Calibration (size n) and Test (size m)
                rng = np.random.default_rng(sim + 1000)
                perm = rng.permutation(N_TOTAL)
                calib_idx = perm[:n]
                test_idx = perm[n:]

                # --- Calibration Step ---
                # Calculate True Ranks within Calibration Set (Rc)
                # These are the ranks of Y_calib relative to each other.
                Rc = mtd.rank_1_to_n(Y_pool[calib_idx])

                # --- Evaluation Prep ---
                # For evaluation, we look at the true ranks of test points relative to the FULL population
                r_true_test = r_true_full[test_idx]

                # --- Algorithms Loop (RA vs VA) ---
                for mode in ["ra", "va"]:
                    # Prepare Inputs
                    if mode == "ra":
                        pred_calib = r_hat_full[calib_idx]  # Predicted Ranks
                        pred_test = r_hat_full[test_idx]
                    else:
                        pred_calib = scores_full[calib_idx]  # Raw Scores
                        pred_test = scores_full[test_idx]

                    # 1. DCR (Exact Method)
                    t_dcr = mtd.dcr_threshold_exact(n, m, Rc, pred_calib, ALPHA, mode, V_sorted)
                    sets_dcr = mtd.prediction_sets_from_threshold(pred_test, t_dcr, mode, N_TOTAL, V_sorted)
                    fcp, rl, sz = mtd.eval_fcp_and_rl(sets_dcr, r_true_test, N_TOTAL)

                    all_results.append({
                        "Dataset": "Synthetic", "Model": "RANKNET", "Score_Type": mode.upper(),
                        "n": n, "m": m, "Algorithm": "DCR", "Trial": sim,
                        "FCP": fcp, "Relative_Length": rl, "Set_Size": sz
                    })

                    # 2. MDCR (Stochastic Method)
                    t_mdcr = mtd.mdcr_threshold_sampling(n, m, Rc, pred_calib, ALPHA, mode, V_sorted, B=B_MDCR, rng=rng)
                    sets_mdcr = mtd.prediction_sets_from_threshold(pred_test, t_mdcr, mode, N_TOTAL, V_sorted)
                    fcp, rl, sz = mtd.eval_fcp_and_rl(sets_mdcr, r_true_test, N_TOTAL)

                    all_results.append({
                        "Dataset": "Synthetic", "Model": "RANKNET", "Score_Type": mode.upper(),
                        "n": n, "m": m, "Algorithm": "MDCR", "Trial": sim,
                        "FCP": fcp, "Relative_Length": rl, "Set_Size": sz
                    })

    # 6. Save Results
    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(CSV_PATH, index=False)
    print(f"\n>>> Success! Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()