"""
KS Test (Kolmogorov-Smirnov Test) - Deep Dive
Understanding statistical significance of drift

What is KS Test?
- Compares two distributions to see if they're significantly different
- Answers: "Are these two distributions statistically different?"
- Output: p-value (if < 0.05, they're different)

Key Concept: CDF (Cumulative Distribution Function)
- CDF answers: "What % of data is ≤ this value?"
- KS test finds the MAXIMUM distance between two CDFs
- If that distance is BIG → distributions are different
- If that distance is SMALL → distributions are similar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from typing import Tuple

def explain_cumulative_distribution():
    """
    Before we run KS test, you MUST understand CDF (Cumulative Distribution Function)
    """
    print("="*80)
    print("UNDERSTANDING CUMULATIVE DISTRIBUTION FUNCTION (CDF)")
    print("="*80)
    print()
    
    # Simple example with 5 values
    data = np.array([10, 20, 30, 40, 50])
    print("Data: [10, 20, 30, 40, 50]")
    print()
    print("CDF interpretation:")
    print("  'What % of data is ≤ this value?'")
    print()
    print("  Value ≤ 10: 20% (1 out of 5)")
    print("  Value ≤ 20: 40% (2 out of 5)")
    print("  Value ≤ 30: 60% (3 out of 5)")
    print("  Value ≤ 40: 80% (4 out of 5)")
    print("  Value ≤ 50: 100% (5 out of 5)")
    print()
    print("Visual representation:")
    print("  100% ────────────────────●")
    print("   80% ──────────────────●")
    print("   60% ────────────────●")
    print("   40% ──────────────●")
    print("   20% ──────────●")
    print("    0% ─────────────────────")
    print("       10   20   30   40   50")
    print()
    print("KS Test asks: If I drew TWO of these curves (one for training, one for current),")
    print("what's the MAXIMUM DISTANCE between them?")
    print()
    print("If that distance is BIG → distributions are different")
    print("If that distance is SMALL → distributions are similar")
    print()
    print("="*80)
    print()


def calculate_ks_test(reference_data: np.ndarray,
                      current_data: np.ndarray,
                      verbose: bool = True) -> Tuple[float, float, str]:
    """
    Perform Kolmogorov-Smirnov Test
    
    The KS test compares two distributions by finding the maximum distance
    between their cumulative distribution functions (CDFs).
    
    Args:
        reference_data: numpy array of baseline values
        current_data: numpy array of current values
        verbose: print detailed info (default True)
    
    Returns:
        ks_statistic: float, the KS test statistic (0 to 1)
        p_value: float, probability that distributions are the same
        interpretation: string, what it means
    
    Key Insight:
        KS Statistic: Maximum distance between two CDFs (ranges 0-1)
        P-value: Probability that these two distributions are actually the same
        
        If p-value < 0.05: We reject the null hypothesis
                          → Distributions ARE significantly different
        If p-value >= 0.05: We fail to reject null hypothesis
                           → Distributions are similar
    """
    
    if verbose:
        print("🧪 Kolmogorov-Smirnov Test")
        print(f"   Reference samples: {len(reference_data)}")
        print(f"   Current samples: {len(current_data)}")
        print()
    
    # Perform KS test
    ks_statistic, p_value = ks_2samp(reference_data, current_data)
    
    if verbose:
        print(f"📊 KS Test Results:")
        print(f"   KS Statistic: {ks_statistic:.6f}")
        print(f"   P-value: {p_value:.6f}")
        print()
        
        # Interpretation
        print(f"📈 Interpretation:")
        if ks_statistic < 0.05:
            print(f"   KS Statistic = {ks_statistic:.4f} (Very small)")
            print(f"   → Maximum distance between CDFs is tiny")
            print(f"   → Distributions look very similar")
        elif ks_statistic < 0.15:
            print(f"   KS Statistic = {ks_statistic:.4f} (Small)")
            print(f"   → Some distance between CDFs")
            print(f"   → Distributions have some differences")
        elif ks_statistic < 0.3:
            print(f"   KS Statistic = {ks_statistic:.4f} (Moderate)")
            print(f"   → Meaningful distance between CDFs")
            print(f"   → Distributions are noticeably different")
        else:
            print(f"   KS Statistic = {ks_statistic:.4f} (Large)")
            print(f"   → Significant distance between CDFs")
            print(f"   → Distributions are very different")
        
        print()
        print(f"📌 P-value = {p_value:.6f}")
        
        if p_value < 0.001:
            print(f"   → P-value is TINY (< 0.001)")
            print(f"   → Extremely strong evidence distributions differ")
        elif p_value < 0.05:
            print(f"   → P-value is small (< 0.05)")
            print(f"   → Strong evidence distributions differ")
        elif p_value < 0.1:
            print(f"   → P-value is moderate (0.05-0.1)")
            print(f"   → Weak evidence distributions differ")
        else:
            print(f"   → P-value is large (> 0.1)")
            print(f"   → No strong evidence distributions differ")
        
        print()
        
        # Decision
        alpha = 0.05  # Standard significance level
        if p_value < alpha:
            interpretation = f"🚨 SIGNIFICANT DIFFERENCE (p < 0.05)"
            explanation = "We reject the null hypothesis. These distributions ARE statistically different."
        else:
            interpretation = f"✅ NO SIGNIFICANT DIFFERENCE (p >= 0.05)"
            explanation = "We fail to reject null hypothesis. Distributions appear similar."
        
        print(f"Decision (α=0.05):")
        print(f"   {interpretation}")
        print(f"   {explanation}")
        print()
    
    return ks_statistic, p_value, interpretation


def visualize_cdf(reference_data, current_data, title):
    """
    Visualize the Cumulative Distribution Functions
    This is what KS test is comparing
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Compute CDFs
    ref_sorted = np.sort(reference_data)
    curr_sorted = np.sort(current_data)
    ref_cdf = np.arange(1, len(ref_sorted) + 1) / len(ref_sorted)
    curr_cdf = np.arange(1, len(curr_sorted) + 1) / len(curr_sorted)
    
    # Plot 1: Both CDFs
    ax1 = axes[0]
    ax1.plot(ref_sorted, ref_cdf, label='Training CDF', linewidth=2, color='blue')
    ax1.plot(curr_sorted, curr_cdf, label='Current CDF', linewidth=2, color='red')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Cumulative Probability')
    ax1.set_title(f'{title}\nCumulative Distribution Functions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Histograms (for reference)
    ax2 = axes[1]
    ax2.hist(reference_data, bins=30, alpha=0.6, label='Training', 
             color='blue', density=True)
    ax2.hist(current_data, bins=30, alpha=0.6, label='Current', 
             color='red', density=True)
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title(f'{title}\nProbability Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# First, explain CDF
explain_cumulative_distribution()

print()

# ============================================================================
# SCENARIO 1: NO DRIFT
# ============================================================================
print("="*80)
print("SCENARIO 1: NO DRIFT - Identical distributions")
print("="*80)
print()

np.random.seed(42)
training_no_drift = np.random.normal(loc=30, scale=10, size=1000)
current_no_drift = np.random.normal(loc=30, scale=10, size=1000)

ks_stat_1, p_val_1, interp_1 = calculate_ks_test(training_no_drift, current_no_drift)

fig1 = visualize_cdf(training_no_drift, current_no_drift, "Scenario 1: No Drift")
plt.savefig('experiments/ks_test_no_drift_cdf.png', dpi=150, bbox_inches='tight')
plt.close()

print()
print()


# ============================================================================
# SCENARIO 2: SMALL DRIFT
# ============================================================================
print("="*80)
print("SCENARIO 2: SMALL DRIFT - Slight shift")
print("="*80)
print()

training_small_drift = np.random.normal(loc=30, scale=10, size=1000)
current_small_drift = np.random.normal(loc=33, scale=10, size=1000)

ks_stat_2, p_val_2, interp_2 = calculate_ks_test(training_small_drift, current_small_drift)

fig2 = visualize_cdf(training_small_drift, current_small_drift, "Scenario 2: Small Drift")
plt.savefig('experiments/ks_test_small_drift_cdf.png', dpi=150, bbox_inches='tight')
plt.close()

print()
print()


# ============================================================================
# SCENARIO 3: SIGNIFICANT DRIFT
# ============================================================================
print("="*80)
print("SCENARIO 3: SIGNIFICANT DRIFT - Major shift")
print("="*80)
print()

training_sig_drift = np.random.normal(loc=30, scale=10, size=1000)
current_sig_drift = np.random.normal(loc=45, scale=10, size=1000)

ks_stat_3, p_val_3, interp_3 = calculate_ks_test(training_sig_drift, current_sig_drift)

fig3 = visualize_cdf(training_sig_drift, current_sig_drift, "Scenario 3: Significant Drift")
plt.savefig('experiments/ks_test_significant_drift_cdf.png', dpi=150, bbox_inches='tight')
plt.close()

print()
print()


# ============================================================================
# COMPARISON: PSI vs KS TEST
# ============================================================================
print("="*80)
print("COMPARISON: PSI vs KS TEST")
print("="*80)
print()

scenarios_data = [
    ("Scenario 1: No Drift", training_no_drift, current_no_drift),
    ("Scenario 2: Small Drift", training_small_drift, current_small_drift),
    ("Scenario 3: Significant Drift", training_sig_drift, current_sig_drift),
]

print(f"{'Scenario':<30} {'KS Stat':<15} {'P-value':<15} {'Significant?':<15}")
print("-" * 75)

for scenario_name, train, curr in scenarios_data:
    ks_stat, p_val, _ = calculate_ks_test(train, curr, verbose=False)
    sig = "YES" if p_val < 0.05 else "NO"
    print(f"{scenario_name:<30} {ks_stat:<15.6f} {p_val:<15.6f} {sig:<15}")

print()
print("Key Differences (PSI vs KS):")
print("  PSI: Continuous scale, tells you MAGNITUDE of shift")
print("  KS Test: Binary decision, tells you if shift is SIGNIFICANT")
print()
print("="*80)
