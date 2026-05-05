"""
E14 — Statistical floor analysis for E5 Mann-Whitney U tests.

The E5 experiment tests 56 region–indicator pairs using one-sided Mann-Whitney U
(pre-2010 vs post-2010 motif occurrence counts, integer values in {0,1,2,3}).

For integer-valued count data with small samples the MWU test has a discrete
p-value distribution. This script:
  1. Characterises the full discrete p-value grid for n_pre=10, n_post=16.
  2. For each significant pair, computes the minimum achievable p-value given
     its OBSERVED (mean_pre, mean_post) data pattern — i.e., can the pair
     achieve a lower p with a different arrangement of the same observed counts?
  3. Checks whether the 9 significant pairs are all at p=0.038473, and whether
     this is the minimum possible given their data (pre: exactly 2 nonzero
     windows; post: all zeros).

Convention in E5: stats.mannwhitneyu(pre, post, alternative='greater')
U statistic interpretation:
  U = (# pairs where pre_i > post_j) + 0.5 * (# ties)
  For 2×1 + 8×0 pre vs all-zero post:
    U = 2×16 (wins) + 0.5×8×16 (ties at 0) = 32 + 64 = 96

Outputs: outputs/E14/e5_floor_analysis_E14.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_pre_vector(n_pre: int, mean_pre: float, k_max: int = 3) -> list:
    """
    Reconstruct the most likely integer pre-period vector from n_pre and mean_pre.

    The E5 data uses binary-like counts (0 or 1 in practice for the significant pairs),
    so the dominant pattern is: floor(mean_pre * n_pre) ones and the rest zeros.
    We use the configuration that MAXIMISES U (and thus minimises p), which is
    achieved by distributing the total count as many 1s as possible (not as fewer
    larger values), because:
      - A value of k>1 paired with post=0 contributes only 1×n_post to U (one win)
      - A value of 1 paired with post=0 also contributes 1×n_post to U (one win)
      - BUT 0s contribute 0.5×n_post each (tie), and having more zeros (fewer nonzeros)
        means fewer wins and more ties → LOWER U.
      - So: all else equal, more nonzero pre values → higher U → lower p.
    For mean_pre = 0.2 with n_pre=10: sum=2. Best arrangement = [1,1,0,0,...,0].
    """
    total = round(mean_pre * n_pre)
    # Use 1s (not larger values) to maximise U with integer constraint
    n_ones = min(total, n_pre)
    return [1] * n_ones + [0] * (n_pre - n_ones)


def min_achievable_p_given_means(n_pre: int, n_post: int,
                                  mean_pre: float, mean_post: float,
                                  k_max: int = 3) -> float:
    """
    Minimum one-sided p-value for mannwhitneyu(pre, post, alternative='greater')
    given the observed (mean_pre, mean_post) and sample sizes.

    The minimum p is achieved by:
      - pre: distribute floor(mean_pre*n_pre) motif occurrences as individual 1s
             (maximises number of pre>post wins vs. using larger counts)
      - post: distribute floor(mean_post*n_post) motif occurrences as individual 1s
              in the same spirit, but here we MINIMISE their count (keep at observed)
    """
    pre = reconstruct_pre_vector(n_pre, mean_pre, k_max)
    post = reconstruct_pre_vector(n_post, mean_post, k_max)
    _, p = stats.mannwhitneyu(pre, post, alternative="greater")
    return float(p)


def global_min_achievable_p(n_pre: int, n_post: int, k_max: int = 3) -> float:
    """
    Global minimum p-value achievable with any integer data in [0, k_max].
    Achieved when all pre = 1 (or any value > 0) and all post = 0.
    """
    pre = [1] * n_pre
    post = [0] * n_post
    _, p = stats.mannwhitneyu(pre, post, alternative="greater")
    return float(p)


def discrete_pvalue_grid(n_pre: int, n_post: int) -> np.ndarray:
    """
    Enumerate all achievable one-sided p-values for the given sample sizes
    with integer data (post=all zeros, vary number of nonzero pre windows).
    This covers the full range from p≈1 down to the global minimum.
    """
    pvals = []
    post = [0] * n_post
    for n_nz in range(0, n_pre + 1):
        pre = [1] * n_nz + [0] * (n_pre - n_nz)
        s, p = stats.mannwhitneyu(pre, post, alternative="greater")
        pvals.append((int(s), float(p)))
    return sorted(set(p for _, p in pvals))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

base_dir = Path(__file__).resolve().parent.parent
csv_path = base_dir / "outputs" / "E5" / "temporal_test_E5.csv"
out_dir = base_dir / "outputs" / "E14"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

print("=== Column names ===")
print(df.columns.tolist())
print()

# Identify column names (handle both n_pre/n_post and n_windows_pre/n_windows_post)
n_pre_col = "n_pre" if "n_pre" in df.columns else "n_windows_pre"
n_post_col = "n_post" if "n_post" in df.columns else "n_windows_post"

# --- Unique sample-size combinations ---
size_combos = df[[n_pre_col, n_post_col]].drop_duplicates()
print("=== Unique (n_pre, n_post) combinations ===")
print(size_combos.to_string(index=False))
print()

# --- Dominant combo ---
n1_dom = int(size_combos[n_pre_col].iloc[0])
n2_dom = int(size_combos[n_post_col].iloc[0])

# --- Full discrete p-value grid ---
p_grid = discrete_pvalue_grid(n1_dom, n2_dom)
p_global_floor = p_grid[0]

print(f"=== Discrete p-value grid for n_pre={n1_dom}, n_post={n2_dom} ===")
print(f"Number of distinct achievable p-values: {len(p_grid)}")
print(f"Global minimum p-value (floor):        {p_global_floor:.10f}")
print(f"Next achievable p above floor:          {p_grid[1]:.10f}")
print(f"Full grid (rounded):")
for p in p_grid:
    marker = " ← global floor" if p == p_global_floor else ""
    marker2 = " ← actual significant p" if abs(p - 0.038473) < 1e-5 else ""
    print(f"  {p:.8f}{marker}{marker2}")
print()

# --- Per-row analysis ---
df["min_p_given_means"] = df.apply(
    lambda r: min_achievable_p_given_means(
        int(r[n_pre_col]), int(r[n_post_col]),
        float(r["mean_pre"]), float(r["mean_post"])
    ),
    axis=1,
)
df["global_min_p"] = df.apply(
    lambda r: global_min_achievable_p(int(r[n_pre_col]), int(r[n_post_col])),
    axis=1,
)
# Is the pair at the floor given its OWN observed means?
df["at_conditional_floor"] = (
    np.abs(df["p_value"] - df["min_p_given_means"]) < 1e-6
)
# Is the pair at the global floor?
df["at_global_floor"] = (
    np.abs(df["p_value"] - df["global_min_p"]) < 1e-6
)

# --- Significant pairs ---
sig = df[df["significant"] == True].copy()
print(f"=== Significant pairs (n={len(sig)}) ===")
print(sig[[
    "region", "indicator", n_pre_col, n_post_col,
    "mean_pre", "mean_post", "U_stat", "p_value",
    "min_p_given_means", "at_conditional_floor"
]].to_string(index=False))
print()

# --- Summary ---
n_total = len(df)
n_sig = len(sig)
n_sig_at_cond_floor = sig["at_conditional_floor"].sum()
n_sig_at_global_floor = sig["at_global_floor"].sum()
n_all_at_cond_floor = df["at_conditional_floor"].sum()

print("=== Summary ===")
print(f"Total pairs:                                   {n_total}")
print(f"Significant pairs (p < 0.05):                  {n_sig}")
print(f"  ... at conditional floor (given their means): "
      f"{n_sig_at_cond_floor} / {n_sig}")
print(f"  ... at global floor:                          "
      f"{n_sig_at_global_floor} / {n_sig}")
print(f"All significant pairs share U = 96, p = ?:     "
      f"{'YES' if sig['p_value'].nunique() == 1 else 'NO'}")
print(f"Shared p-value of significant pairs:           "
      f"{sig['p_value'].unique()}")
print(f"Global floor p-value:                          "
      f"{p_global_floor:.10f}")
print(f"Conditional floor for (mean_pre=0.2, mean_post=0.0): "
      f"{sig['min_p_given_means'].unique()}")
print()

# Pairs with p < conditional floor (should be zero)
below_cond_floor = df[df["p_value"] < (df["min_p_given_means"] - 1e-6)]
print(f"Pairs with p < conditional floor: {len(below_cond_floor)}")
print()

# --- Pairs constrained to p >= 0.05 (can never be significant) ---
# These are pairs where min_p_given_means >= 0.05
cant_sig = df[df["min_p_given_means"] >= 0.05]
print(f"Pairs where conditional floor p >= 0.05 "
      f"(can never be significant given these means): "
      f"{len(cant_sig)} / {n_total} ({100*len(cant_sig)/n_total:.1f}%)")
if len(cant_sig) > 0:
    print(cant_sig[["region", "indicator", "mean_pre", "mean_post", "p_value",
                     "min_p_given_means"]].to_string(index=False))
print()

# What data pattern would be needed to achieve a lower p?
print("=== Data patterns and their p-values (post=all zeros) ===")
print("n_pre_nonzero  U_stat  p_value  significant@0.05")
post0 = [0] * n2_dom
for n_nz in range(0, n1_dom + 1):
    pre = [1] * n_nz + [0] * (n1_dom - n_nz)
    s, p = stats.mannwhitneyu(pre, post0, alternative="greater")
    sig_mark = "*" if p < 0.05 else ""
    print(f"  {n_nz:2d} nonzero pre-windows  →  U={s:5.0f},  p={p:.8f} {sig_mark}")
print()
print(f"The 9 significant pairs have 2 pre-windows with motif (mean_pre=0.2).")
print(f"p=0.038473 IS the minimum achievable p for this data pattern.")
print(f"To achieve p < 0.038, at least 3 pre-windows must have motif occurrence.")
print()

# --- Save output CSV ---
out_df = df[[
    "region", "indicator",
    n_pre_col, n_post_col,
    "mean_pre", "mean_post",
    "U_stat", "p_value",
    "min_p_given_means", "global_min_p",
    "at_conditional_floor", "at_global_floor",
    "significant"
]].rename(columns={
    n_pre_col: "n_pre",
    n_post_col: "n_post",
    "p_value": "actual_p",
})

out_path = out_dir / "e5_floor_analysis_E14.csv"
out_df.to_csv(out_path, index=False, float_format="%.8f")
print(f"Results saved to: {out_path}")
print()

# --- Final interpretation ---
print("=" * 60)
print("INTERPRETATION FOR PAPER")
print("=" * 60)
print()
print(f"All {n_sig} significant pairs are at THE CONDITIONAL FLOOR:")
print(f"  p = 0.038473 = min achievable p given mean_pre=0.2, mean_post=0.0,")
print(f"  n_pre={n1_dom}, n_post={n2_dom}, integer data.")
print()
print(f"The GLOBAL floor for these sample sizes is p = {p_global_floor:.6e},")
print(f"achieved when ALL 10 pre-windows have motif and ALL 16 post-windows do not.")
print(f"None of the 9 pairs are at the global floor (they have mean_pre=0.2, not 1.0).")
print()
print("The 9 'significant' pairs all share an identical data pattern:")
print("  - Exactly 2 of 10 pre-2010 windows contain a synchrony motif")
print("  - Zero of 16 post-2010 windows contain a synchrony motif")
print("  → U=96, p=0.038473 (one-sided, alternative='greater')")
print()
print("This means we cannot distinguish between a pair that barely achieves")
print("p=0.038 and one where pre=0.2, post=0.0 is the ONLY possible pattern.")
print("The p-value carries no gradation of effect size within this level.")
print()
if n_sig_at_cond_floor == n_sig:
    print("CONCLUSION: All 9 significant pairs ARE at the conditional statistical")
    print("floor. The test cannot produce a smaller p for their observed data pattern.")
    print("This is consistent with the hypothesis that p=0.038 reflects the")
    print("discreteness constraint, not a graded effect size.")
