"""
Jensen-Shannon Divergence - Deep Dive
Flexible divergence measure for any distribution

What is Jensen-Shannon Divergence?
- Measures the difference between two probability distributions
- Works for BOTH continuous AND categorical data
- Based on KL (Kullback-Leibler) divergence, but symmetric
- Output: Value between 0 and 1
  * 0 = distributions are identical
  * 1 = distributions are completely different
  * 0.5 = distributions are somewhat similar

Why "Jensen-Shannon" and not just KL divergence?
- KL divergence is asymmetric (KL(P||Q) ≠ KL(Q||P))
- JS makes it symmetric: JS(P||Q) = JS(Q||P)
- More intuitive for comparing distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from typing import Tuple

def explain_kl_divergence():
    """
    Explain KL divergence and why Jensen-Shannon is better
    """
    print("="*80)
    print("UNDERSTANDING KULLBACK-LEIBLER (KL) DIVERGENCE")
    print("="*80)
    print()
    
    print("KL Divergence asks: How much does distribution P differ from Q?")
    print()
    print("Simple example with coin flips:")
    print("  Distribution P (Fair coin): 50% heads, 50% tails")
    print("  Distribution Q (Biased coin): 60% heads, 40% tails")
    print()
    print("KL(P||Q) = How surprised would P be if it saw Q?")
    print("           Answer: 0.0163 nats (slightly surprised)")
    print()
    print("KL(Q||P) = How surprised would Q be if it saw P?")
    print("           Answer: 0.0170 nats (slightly more surprised)")
    print()
    print("Notice: KL(P||Q) ≠ KL(Q||P) → KL is ASYMMETRIC")
    print()
    print("Problem: Which direction should we use?")
    print("Solution: Use Jensen-Shannon (symmetric average)")
    print()
    print("="*80)
    print()


def explain_jensen_shannon():
    """
    Explain why Jensen-Shannon is the better choice
    """
    print("="*80)
    print("JENSEN-SHANNON DIVERGENCE: THE SOLUTION")
    print("="*80)
    print()
    
    print("Jensen-Shannon is the AVERAGE of KL divergence in both directions:")
    print()
    print("JS(P||Q) = 0.5 * KL(P||Q) + 0.5 * KL(Q||P)")
    print()
    print("Advantages:")
    print("  ✓ Symmetric: JS(P||Q) = JS(Q||P)")
    print("  ✓ Bounded: Always between 0 and 1")
    print("  ✓ Interpretable: Easy to understand")
    print("  ✓ Flexible: Works for continuous and categorical")
    print()
    print("Interpretation:")
    print("  JS = 0.0  → Distributions are identical")
    print("  JS = 0.1  → Distributions are very similar")
    print("  JS = 0.3  → Distributions have noticeable differences")
    print("  JS = 0.5  → Distributions are moderately different")
    print("  JS = 0.8  → Distributions are very different")
    print("  JS = 1.0  → Distributions are completely different")
    print()
    print("="*80)
    print()


def calculate_jensen_shannon(reference_data: np.ndarray,
                             current_data: np.ndarray,
                             bins: int = 10,
                             verbose: bool = True) -> Tuple[float, str]:
    """
    Calculate Jensen-Shannon Divergence between two distributions
    
    Args:
        reference_data: numpy array of baseline values
        current_data: numpy array of current values
        bins: number of bins for histogram (default 10)
        verbose: print detailed info (default True)
    
    Returns:
        js_divergence: float, the Jensen-Shannon divergence (0-1)
        interpretation: string, what it means
    
    Process:
        1. Create histograms (probability distributions)
        2. Normalize to probabilities (sum to 1)
        3. Calculate JS divergence
    """
    
    if verbose:
        print("📊 Jensen-Shannon Divergence")
        print(f"   Reference samples: {len(reference_data)}")
        print(f"   Current samples: {len(current_data)}")
        print()
    
    # Step 1: Create histograms (get counts for each bin)
    breakpoints = np.percentile(reference_data, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    ref_counts = np.histogram(reference_data, bins=breakpoints)[0]
    curr_counts = np.histogram(current_data, bins=breakpoints)[0]
    
    # Step 2: Normalize to probabilities (convert to percentages)
    ref_probs = ref_counts / np.sum(ref_counts)
    curr_probs = curr_counts / np.sum(curr_counts)
    
    # Step 3: Handle zero probabilities
    # JS divergence can handle zeros, but adding epsilon prevents numerical issues
    epsilon = 1e-10
    ref_probs = np.where(ref_probs == 0, epsilon, ref_probs)
    curr_probs = np.where(curr_probs == 0, epsilon, curr_probs)
    
    # Re-normalize after adding epsilon
    ref_probs = ref_probs / np.sum(ref_probs)
    curr_probs = curr_probs / np.sum(curr_probs)
    
    # Step 4: Calculate Jensen-Shannon Divergence
    js_divergence = jensenshannon(ref_probs, curr_probs)
    
    if verbose:
        print(f"📈 Results:")
        print(f"   Jensen-Shannon Divergence: {js_divergence:.6f}")
        print()
        
        # Interpretation
        if js_divergence < 0.05:
            print(f"   JS = {js_divergence:.4f} (Very small)")
            print(f"   → Distributions are nearly identical")
        elif js_divergence < 0.15:
            print(f"   JS = {js_divergence:.4f} (Small)")
            print(f"   → Distributions are very similar")
        elif js_divergence < 0.3:
            print(f"   JS = {js_divergence:.4f} (Moderate)")
            print(f"   → Distributions have noticeable differences")
        elif js_divergence < 0.5:
            print(f"   JS = {js_divergence:.4f} (Large)")
            print(f"   → Distributions are quite different")
        else:
            print(f"   JS = {js_divergence:.4f} (Very large)")
            print(f"   → Distributions are very different")
        
        print()
        
        # Decision based on typical threshold (0.1)
        if js_divergence < 0.1:
            interpretation = f"✅ SIMILAR DISTRIBUTIONS (JS < 0.1)"
        elif js_divergence < 0.3:
            interpretation = f"⚠️  MODERATE DIFFERENCE (0.1 ≤ JS < 0.3)"
        else:
            interpretation = f"🚨 SIGNIFICANT DIFFERENCE (JS ≥ 0.3)"
        
        print(f"Decision (typical threshold JS=0.1-0.3):")
        print(f"   {interpretation}")
        print()
    
    return js_divergence, interpretation


def visualize_js_comparison(reference_data, current_data, title):
    """
    Visualize distributions being compared by JS divergence
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overlapping histograms
    ax1 = axes[0]
    ax1.hist(reference_data, bins=30, alpha=0.6, label='Training', 
             color='blue', density=True)
    ax1.hist(current_data, bins=30, alpha=0.6, label='Current', 
             color='red', density=True)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{title}\nProbability Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plots for comparison
    ax2 = axes[1]
    data_to_plot = [reference_data, current_data]
    bp = ax2.boxplot(data_to_plot, labels=['Training', 'Current'], patch_artist=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Value')
    ax2.set_title(f'{title}\nDistribution Summary Statistics')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# First, explain the concept
explain_kl_divergence()
print()
explain_jensen_shannon()
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

js_1, interp_1 = calculate_jensen_shannon(training_no_drift, current_no_drift)

fig1 = visualize_js_comparison(training_no_drift, current_no_drift, "Scenario 1: No Drift")
plt.savefig('experiments/js_no_drift.png', dpi=150, bbox_inches='tight')
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

js_2, interp_2 = calculate_jensen_shannon(training_small_drift, current_small_drift)

fig2 = visualize_js_comparison(training_small_drift, current_small_drift, "Scenario 2: Small Drift")
plt.savefig('experiments/js_small_drift.png', dpi=150, bbox_inches='tight')
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

js_3, interp_3 = calculate_jensen_shannon(training_sig_drift, current_sig_drift)

fig3 = visualize_js_comparison(training_sig_drift, current_sig_drift, "Scenario 3: Significant Drift")
plt.savefig('experiments/js_significant_drift.png', dpi=150, bbox_inches='tight')
plt.close()

print()
print()


# ============================================================================
# COMPARISON: ALL 4 DETECTORS
# ============================================================================
print("="*80)
print("JENSEN-SHANNON SUMMARY: WHY IT'S SPECIAL")
print("="*80)
print()

scenarios_data = [
    ("Scenario 1: No Drift", training_no_drift, current_no_drift),
    ("Scenario 2: Small Drift", training_small_drift, current_small_drift),
    ("Scenario 3: Significant Drift", training_sig_drift, current_sig_drift),
]

print(f"{'Scenario':<30} {'JS Divergence':<20} {'Interpretation':<30}")
print("-" * 80)

for scenario_name, train, curr in scenarios_data:
    js_val, interp = calculate_jensen_shannon(train, curr, verbose=False)
    if js_val < 0.1:
        status = "Similar"
    elif js_val < 0.3:
        status = "Moderate Diff"
    else:
        status = "Major Diff"
    print(f"{scenario_name:<30} {js_val:<20.6f} {status:<30}")

print()
print("Key Advantages of Jensen-Shannon:")
print("  ✓ Symmetric (doesn't matter which direction)")
print("  ✓ Bounded (0-1 scale, easy to interpret)")
print("  ✓ Works for any distribution (continuous or categorical)")
print("  ✓ No special requirements (handles edge cases well)")
print()
print("When to Use Jensen-Shannon:")
print("  ✓ You want one metric that works for everything")
print("  ✓ You're comparing both continuous and categorical features")
print("  ✓ You want a symmetric distance measure")
print()
print("="*80)
