# KS Test Learning Answers

## Q1: What is a CDF? Explain in your own words.
A CDF (Cumulative Distribution Function) answers: 'What percentage of my data is ≤ this value?'
It's a curve that goes from 0% to 100%, showing how much data falls below each point.

## Q2: In Scenario 1 (No Drift), what was the KS statistic? Why would the KS test say 'no significant difference'?
KS Statistic: 0.045
P-value: 0.263472
The distance between the two CDFs is tiny (0.045), and the p-value (0.26) is >> 0.05, 
so we fail to reject the null hypothesis. The distributions appear identical.

## Q3: In Scenario 3 (Significant Drift), what was the p-value? What does that p-value tell us?
P-value: 0.000000 (essentially 0)
This tiny p-value means there's virtually NO chance these distributions are the same if we observed this much difference by random variation.
We REJECT the null hypothesis. Distributions are statistically significantly different.

## Q4: PSI and KS measure different things. Explain the difference.
- PSI: Tells you HOW MUCH the distribution shifted (continuous scale, 0 to ∞)
- KS Test: Tells you IF the shift is STATISTICALLY SIGNIFICANT (binary with p-value)
PSI = magnitude, KS = significance.

## Q5: If you had to choose ONE test (PSI or KS) for production monitoring, which would you choose? Why?
I'd choose BOTH:
- Use PSI to track the MAGNITUDE of drift over time (trending)
- Use KS Test for ALERTING when drift exceeds significance threshold
Together they give complete picture: what's drifting AND when it matters enough to act on.

