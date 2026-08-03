# KS Test (Kolmogorov-Smirnov Test) - Learning Document

## What is KS Test?

The Kolmogorov-Smirnov test is a statistical test that answers one question:

**"Are these two distributions statistically significantly different?"**

Key insight: It's a **BINARY** answer (YES or NO), not a continuous scale like PSI.

---

## The Math Behind CDF

Before understanding KS, you need to understand **CDF (Cumulative Distribution Function)**.

### What is a CDF?

A CDF answers: **"What percentage of my data is ≤ this value?"**

**Example:**
```
Data: [10, 20, 30, 40, 50]

CDF at 10: 20% (1 out of 5 values)
CDF at 20: 40% (2 out of 5 values)
CDF at 30: 60% (3 out of 5 values)
CDF at 40: 80% (4 out of 5 values)
CDF at 50: 100% (5 out of 5 values)
```

### Visualizing CDF

```
100% ────────────────────●
 80% ──────────────────●
 60% ────────────────●
 40% ──────────────●
 20% ──────────●
  0% ─────────────────────
     10   20   30   40   50
```

---

## How KS Test Works

### Step 1: Build CDFs for Both Datasets

- Training data → CDF1
- Current data → CDF2

### Step 2: Find Maximum Distance

Find the point where CDF1 and CDF2 are furthest apart.

**Example:**
```
At value 25:
  CDF1 (training): 50%
  CDF2 (current):  30%
  Distance: |50% - 30%| = 20%

At value 35:
  CDF1 (training): 65%
  CDF2 (current):  75%
  Distance: |65% - 75%| = 10%

Maximum distance (KS Statistic) = 20%
```

### Step 3: Calculate P-value

The p-value answers: **"If these distributions were actually the same, what's the probability we'd see this large a distance by random chance?"**

---

## Interpretation

### KS Statistic (0 to 1)

- **KS < 0.05**: Distributions are very similar
- **KS 0.05-0.15**: Some difference, but still close
- **KS 0.15-0.3**: Noticeable difference
- **KS > 0.3**: Very different distributions

### P-value

- **p-value < 0.05**: Strong evidence distributions are different (REJECT null hypothesis)
- **p-value ≥ 0.05**: Weak evidence distributions are different (FAIL TO REJECT)

**Standard threshold: α = 0.05**

---

## PSI vs KS Test

| Aspect | PSI | KS Test |
|--------|-----|---------|
| **Output** | Continuous (0 to ∞) | Continuous (0 to 1) |
| **What it measures** | Magnitude of shift | Statistical significance |
| **Question answered** | By how much did it shift? | Is it significantly different? |
| **Use case** | Track drift over time | Make binary alert decision |
| **Threshold** | PSI > 0.25 = alert | p-value < 0.05 = alert |

### Example Scenario

```
Scenario: Customer age distribution changed
- Training mean: 35 years
- Current mean: 38 years

PSI = 0.12 (small magnitude shift)
KS p-value = 0.02 (statistically significant)

Interpretation:
- PSI says: "The shift is small, but consistent"
- KS says: "This shift is unlikely to happen by chance"

Decision: Monitor, might need retraining
```

---

## When to Use KS Test

✅ Use KS Test when:
- You need a BINARY decision (drift or no drift)
- You're setting automated alerts
- You want to know if the shift is statistically significant

❌ Don't use alone when:
- You want to understand the MAGNITUDE of drift
- You need to explain "how much" changed
- You're tracking drift over a timeline

---

## Real-World Example

**Model:** Fraud detection  
**Feature:** Transaction amount (continuous)

**Training data distribution:**
```
Mean: $100
Std Dev: $50
Range: $10 - $500
```

**After 1 month:**
```
Mean: $105
Std Dev: $52
Range: $10 - $520
```

**Results:**
```
KS Statistic: 0.03
P-value: 0.45

Interpretation: 
- Very small KS statistic (0.03)
- High p-value (0.45 > 0.05)
- Conclusion: NOT statistically significant drift
- Action: No alert needed
```

**After 6 months:**
```
Mean: $200
Std Dev: $100
Range: $20 - $1000
```

**Results:**
```
KS Statistic: 0.42
P-value: 0.000001

Interpretation:
- Large KS statistic (0.42)
- Tiny p-value (0.000001 << 0.05)
- Conclusion: HIGHLY statistically significant drift
- Action: ALERT - trigger retraining pipeline
```

---

## Questions After Running the Script

### Q1: What is a CDF? Explain in your own words.
> Answer here after running the script

### Q2: In Scenario 1 (No Drift), what was the KS statistic? Why would the KS test say "no significant difference"?
> Answer here after running the script

### Q3: In Scenario 3 (Significant Drift), what was the p-value? What does that p-value tell us?
> Answer here after running the script

### Q4: PSI and KS measure different things. Explain the difference.
> Answer here after running the script

### Q5: If you had to choose ONE test (PSI or KS) for your production monitoring, which would you choose? Why?
> Answer here after running the script

---

## Key Takeaway

**KS Test is the "significance checker"**

- PSI tells you HOW MUCH
- KS Test tells you IF IT MATTERS (statistically)

Use them together for complete drift monitoring.
