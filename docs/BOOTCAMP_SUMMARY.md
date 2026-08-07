# DriftWatch Statistical Foundations Bootcamp - Complete Summary

## Overview
This bootcamp covers 4 essential drift detection methods for ML model monitoring.

## Methods Learned

### 1. PSI (Population Stability Index)
- **Measure**: Magnitude of distribution shift
- **Formula**: Σ (current% - ref%) × ln(current% / ref%)
- **Range**: 0 to ∞
- **Threshold**: PSI > 0.25 = alert
- **Best For**: Tracking drift magnitude over time

### 2. KS Test (Kolmogorov-Smirnov)
- **Measure**: Statistical significance of distribution difference
- **Method**: Compare CDFs, find maximum distance
- **Output**: p-value (0 to 1)
- **Threshold**: p-value < 0.05 = significant drift
- **Best For**: Binary alert decisions

### 3. Chi-Squared Test
- **Measure**: Categorical distribution shift
- **Method**: Compare contingency tables
- **Output**: χ² statistic and p-value
- **Best For**: Product categories, regions, segments

### 4. Jensen-Shannon Divergence
- **Measure**: Flexible, symmetric divergence
- **Formula**: Symmetric average of KL divergence
- **Range**: 0 to 1 (bounded, interpretable)
- **Best For**: Mixed data types, balanced comparison

## Files Created
- experiments/02_ks_test_deep_dive.py
- experiments/03_chi_squared_deep_dive.py
- experiments/04_jensen_shannon_deep_dive.py
- docs/BOOTCAMP_KS_TEST.md
- docs/BOOTCAMP_CHI2.md
- docs/BOOTCAMP_JENSEN_SHANNON.md
- docs/BOOTCAMP_KS_TEST_ANSWERS.md

## Visualizations Generated
- CDF comparison charts (KS Test)
- Contingency table breakdowns (Chi²)
- Distribution density plots (Jensen-Shannon)

## Production Implementation Strategy
1. **Data Ingestion**: CSV/Parquet upload or Kafka stream
2. **Baseline Creation**: Calculate reference distributions
3. **Drift Detection**: Run all 4 tests in parallel
4. **Alerting**: PSI for trending, KS/Chi²/JS for thresholds
5. **Action**: Trigger retraining pipeline when drift confirmed

## Key Insights
- No single test is perfect; use them together
- Statistical significance ≠ practical importance
- Always visualize before trusting numbers
- Combine magnitude (PSI) with significance (KS) for robust monitoring

