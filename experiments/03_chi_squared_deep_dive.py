"""
Chi-Squared Test - Deep Dive
Testing for categorical feature distribution drift

What is Chi-Squared Test?
- Tests if two categorical distributions are significantly different
- For features like: product_category, region, user_segment, payment_method
- Cannot use KS test for categorical data (KS only works for continuous)
- Output: p-value (if < 0.05, distributions are different)

Key Concept: Contingency Table
- Shows observed vs expected counts for each category
- Chi² measures how far actual counts deviate from expected
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency


def explain_contingency_table():
    """
    Explain what a contingency table is and why we use it for Chi-Squared
    """
    print("=" * 80)
    print("UNDERSTANDING CONTINGENCY TABLES FOR CHI-SQUARED TEST")
    print("=" * 80)
    print()

    print("Scenario: E-commerce product category distribution")
    print()
    print("Training Data (Historical):")
    print("  Electronics: 500 customers (50%)")
    print("  Clothing: 300 customers (30%)")
    print("  Books: 150 customers (15%)")
    print("  Other: 50 customers (5%)")
    print("  TOTAL: 1000 customers")
    print()

    print("Current Data (Recent):")
    print("  Electronics: 400 customers (40%)")
    print("  Clothing: 450 customers (36%)")
    print("  Books: 100 customers (8%)")
    print("  Other: 50 customers (4%)")
    print("  TOTAL: 1000 customers")
    print()

    print("Contingency Table (Observed Counts):")
    print("-" * 50)
    print(f"{'Category':<15} {'Training':<15} {'Current':<15}")
    print("-" * 50)
    print(f"{'Electronics':<15} {500:<15} {400:<15}")
    print(f"{'Clothing':<15} {300:<15} {450:<15}")
    print(f"{'Books':<15} {150:<15} {100:<15}")
    print(f"{'Other':<15} {50:<15} {50:<15}")
    print("-" * 50)
    print(f"{'TOTAL':<15} {1000:<15} {1000:<15}")
    print()

    print("Chi-Squared asks: How much do these distributions differ?")
    print("If difference is BIG → p-value < 0.05 → distributions are different")
    print()
    print("=" * 80)
    print()


def calculate_chi2_test(
    reference_counts: np.ndarray,
    current_counts: np.ndarray,
    categories: list,
    verbose: bool = True,
) -> tuple[float, float, str]:
    """
    Perform Chi-Squared Test for categorical distributions

    Args:
        reference_counts: array of counts for each category in training data
        current_counts: array of counts for each category in current data
        categories: list of category names
        verbose: print detailed info (default True)

    Returns:
        chi2_statistic: float, the chi-squared test statistic
        p_value: float, probability that distributions are the same
        interpretation: string, what it means

    Formula:
        χ² = Σ (observed - expected)² / expected

        For each category:
          - Observed: actual count in current data
          - Expected: what we'd expect if distributions were identical
    """

    # Build contingency table
    contingency_table = np.array([reference_counts, current_counts])

    if verbose:
        print("🔍 Chi-Squared Test for Categorical Data")
        print(f"   Categories: {len(categories)}")
        print(f"   Training samples: {np.sum(reference_counts)}")
        print(f"   Current samples: {np.sum(current_counts)}")
        print()

        print("Contingency Table:")
        print("-" * 70)
        print(f"{'Category':<20} {'Training Count':<20} {'Current Count':<20}")
        print("-" * 70)
        for i, cat in enumerate(categories):
            print(f"{cat:<20} {reference_counts[i]:<20} {current_counts[i]:<20}")
        print("-" * 70)
        print()

    # Perform chi-squared test
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

    if verbose:
        print("📊 Chi-Squared Test Results:")
        print(f"   Chi-Squared Statistic: {chi2_stat:.6f}")
        print(f"   P-value: {p_value:.6f}")
        print(f"   Degrees of Freedom: {dof}")
        print()

        print("Expected Frequencies (if distributions were identical):")
        print("-" * 70)
        print(f"{'Category':<20} {'Training Expected':<20} {'Current Expected':<20}")
        print("-" * 70)
        for i, cat in enumerate(categories):
            print(
                f"{cat:<20} {expected_freq[0][i]:<20.2f} {expected_freq[1][i]:<20.2f}"
            )
        print("-" * 70)
        print()

        # Interpretation
        print("📈 Interpretation:")
        if chi2_stat < 1:
            print(f"   χ² = {chi2_stat:.4f} (Very small)")
            print("   → Categories have similar distributions")
        elif chi2_stat < 5:
            print(f"   χ² = {chi2_stat:.4f} (Small)")
            print("   → Some difference in distributions")
        elif chi2_stat < 10:
            print(f"   χ² = {chi2_stat:.4f} (Moderate)")
            print("   → Noticeable difference in distributions")
        else:
            print(f"   χ² = {chi2_stat:.4f} (Large)")
            print("   → Significant difference in distributions")

        print()
        print(f"📌 P-value = {p_value:.6f}")

        if p_value < 0.001:
            print("   → Extremely strong evidence distributions differ")
        elif p_value < 0.05:
            print("   → Strong evidence distributions differ")
        elif p_value < 0.1:
            print("   → Weak evidence distributions differ")
        else:
            print("   → No strong evidence distributions differ")

        print()

        # Decision
        alpha = 0.05
        if p_value < alpha:
            interpretation = "🚨 SIGNIFICANT DIFFERENCE (p < 0.05)"
            explanation = "Categorical distributions are significantly different."
        else:
            interpretation = "✅ NO SIGNIFICANT DIFFERENCE (p >= 0.05)"
            explanation = "Categorical distributions are similar."

        print("Decision (α=0.05):")
        print(f"   {interpretation}")
        print(f"   {explanation}")
        print()

    return chi2_stat, p_value, interpretation


def visualize_categorical_dist(reference_counts, current_counts, categories, title):
    """
    Visualize categorical distributions as bar charts
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate percentages
    ref_pct = (reference_counts / np.sum(reference_counts)) * 100
    curr_pct = (current_counts / np.sum(current_counts)) * 100

    # Plot 1: Side-by-side bar chart
    ax1 = axes[0]
    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width / 2, ref_pct, width, label="Training", color="blue", alpha=0.7)
    ax1.bar(x + width / 2, curr_pct, width, label="Current", color="red", alpha=0.7)

    ax1.set_xlabel("Category")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title(f"{title}\nCategorical Distribution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Difference chart
    ax2 = axes[1]
    diff = curr_pct - ref_pct
    colors = ["red" if d > 0 else "blue" for d in diff]

    ax2.bar(categories, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
    ax2.set_xlabel("Category")
    ax2.set_ylabel("Percentage Difference (%)")
    ax2.set_title(f"{title}\nDistribution Shift by Category")
    ax2.set_xticklabels(categories, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# First, explain contingency tables
explain_contingency_table()

print()

# ============================================================================
# SCENARIO 1: NO DRIFT (Categorical)
# ============================================================================
print("=" * 80)
print("SCENARIO 1: NO DRIFT - Identical categorical distribution")
print("=" * 80)
print()

np.random.seed(42)

# Training distribution
training_categories_1 = np.array(
    [500, 300, 150, 50]
)  # Electronics, Clothing, Books, Other
current_categories_1 = np.array([500, 300, 150, 50])  # Identical

categories = ["Electronics", "Clothing", "Books", "Other"]

chi2_stat_1, p_val_1, interp_1 = calculate_chi2_test(
    training_categories_1, current_categories_1, categories
)

fig1 = visualize_categorical_dist(
    training_categories_1, current_categories_1, categories, "Scenario 1: No Drift"
)
plt.savefig("experiments/chi2_no_drift.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print()


# ============================================================================
# SCENARIO 2: SMALL DRIFT (Categorical)
# ============================================================================
print("=" * 80)
print("SCENARIO 2: SMALL DRIFT - Slight categorical shift")
print("=" * 80)
print()

training_categories_2 = np.array([500, 300, 150, 50])
current_categories_2 = np.array([480, 320, 160, 40])  # Small shifts

chi2_stat_2, p_val_2, interp_2 = calculate_chi2_test(
    training_categories_2, current_categories_2, categories
)

fig2 = visualize_categorical_dist(
    training_categories_2, current_categories_2, categories, "Scenario 2: Small Drift"
)
plt.savefig("experiments/chi2_small_drift.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print()


# ============================================================================
# SCENARIO 3: SIGNIFICANT DRIFT (Categorical)
# ============================================================================
print("=" * 80)
print("SCENARIO 3: SIGNIFICANT DRIFT - Major categorical shift")
print("=" * 80)
print()

training_categories_3 = np.array([500, 300, 150, 50])
current_categories_3 = np.array(
    [400, 450, 100, 50]
)  # Major shift in Electronics/Clothing

chi2_stat_3, p_val_3, interp_3 = calculate_chi2_test(
    training_categories_3, current_categories_3, categories
)

fig3 = visualize_categorical_dist(
    training_categories_3,
    current_categories_3,
    categories,
    "Scenario 3: Significant Drift",
)
plt.savefig("experiments/chi2_significant_drift.png", dpi=150, bbox_inches="tight")
plt.close()

print()
print()


# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("=" * 80)
print("CHI-SQUARED TEST SUMMARY")
print("=" * 80)
print()

scenarios_data = [
    ("Scenario 1: No Drift", training_categories_1, current_categories_1),
    ("Scenario 2: Small Drift", training_categories_2, current_categories_2),
    ("Scenario 3: Significant Drift", training_categories_3, current_categories_3),
]

print(f"{'Scenario':<30} {'χ² Statistic':<20} {'P-value':<15} {'Significant?':<15}")
print("-" * 80)

for scenario_name, train, curr in scenarios_data:
    chi2_stat, p_val, _ = calculate_chi2_test(train, curr, categories, verbose=False)
    sig = "YES" if p_val < 0.05 else "NO"
    print(f"{scenario_name:<30} {chi2_stat:<20.6f} {p_val:<15.6f} {sig:<15}")

print()
print("Key Insight:")
print("  Chi-Squared test is specifically designed for CATEGORICAL features")
print("  Use it when your feature has discrete categories, not continuous values")
print()
print("=" * 80)
