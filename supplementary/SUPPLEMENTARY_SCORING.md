# Supplementary Materials: Benchmark Scoring System

## Overview

We developed a quantitative scoring system to evaluate LLM calibration across our experimental battery. The benchmark assigns a total of 100 points across 5 experiments, with each experiment contributing 20 points. Higher scores indicate better calibration to requested probability distributions.

## Step-Likeness Metric (S)

For experiments involving binary or multi-outcome probability requests, we define the step-likeness metric S to quantify deviation from ideal calibration:

$$S = 4 \times \int_0^1 |r(p) - p| \, dp$$

where r(p) is the observed response rate for a given requested probability p. The factor of 4 normalizes S such that a perfect step function (r = 0 for p < 0.5, r = 1 for p ≥ 0.5) yields S = 1, while perfect calibration (r = p for all p) yields S = 0.

We compute this integral via the trapezoidal rule over experimentally sampled p values, with boundary extrapolation to p = 0 and p = 1.

## Experiment 1: Single Binary Flip

**Design**: For each p ∈ {0%, 5%, 10%, ..., 100%}, we request N = 100 independent binary outputs with the specified probability of returning "1".

**Scoring**:
$$\text{Score}_1 = \max(0, 20 \times (1 - S))$$

A perfectly calibrated model achieves S = 0 and scores 20/20. A perfect step function achieves S = 1 and scores 0/20.

## Experiment 2: Two Sequential Flips

**Design**: Same probability range as Exp1, but requesting two independent flips per API call (D = 2).

**Metrics**: We compute three S values:
- S₁: Step-likeness of first response (j = 1)
- S₂: Step-likeness of second response (j = 2)  
- S_avg: Step-likeness of mean response rate across both flips

**Scoring**:
$$\text{Score}_2 = \max(0, 10 \times (1 - S_2)) + \max(0, 10 \times (1 - S_{\text{avg}}))$$

We report S₁ but do not include it in the score, as S₁ is expected to remain near 1 (step-like) while S₂ and S_avg capture the model's attempted self-correction behavior.

## Experiment 3: Three-Outcome Distribution

**Design**: We fix P("1") = 40% and vary P("0") from 60% to 0% across 7 conditions, with P("2") = 60% - P("0").

**Metrics**: Three separate S values, each normalized to [0, 1]:

$$S_0 = \frac{\int |r_0 - P_0| \, dP_0}{\text{max\_area}_0}$$

$$S_1 = \frac{\int |r_1 - 0.4| \, dP_2}{\text{max\_area}_1}$$

$$S_2 = \frac{\int |r_2 - P_2| \, dP_2}{\text{max\_area}_2}$$

For diagonal expectations (S₀, S₂), max_area accounts for the asymmetric [0, 0.6] range. For the horizontal expectation S₁ (expected r₁ = 0.4), max_area = 0.6 × max(0.4, 0.6) = 0.36.

**Scoring**:
$$\text{Score}_3 = \frac{20}{3} \times [\max(0, 1-S_0) + \max(0, 1-S_1) + \max(0, 1-S_2)]$$

## Experiment 4: Decision Bias

**Design**: Three real-world decision scenarios with mixed-strategy optimal solutions:

1. **Bioinformatics** (8 points): Ambiguous read mapping problem
2. **Asymmetric Matching Pennies** (8 points): Game theory scenario  
3. **Business Positioning** (4 points): Strategic decision problem

**Scoring by scenario**:

*Bioinformatics*: Tests calibration only.
$$\text{Score}_{\text{Bio}} = \max(0, 1 - S) \times 8$$

*Matching Pennies*: Tests both calibration and directional response.
$$\text{Score}_{\text{MP}} = \max(0, 1 - S) \times 4 + d \times 4$$

where d is the direction score: for each adjacent pair of p values, we assign +1 if r increases with p, -1 if r decreases, and 0 otherwise. The raw direction score (mean across pairs) is normalized to [0, 1].

*Luxury vs. Affordable*: Tests directional response only.
$$\text{Score}_{\text{Lux}} = d \times 4$$

**Total Exp4 Score**: Sum of all scenario scores (max 20).

## Experiment 5: Semantic Preference Bias

**Design**: 11 word pairs tested at p = 50%, each in forward and reverse order (22 conditions total, N = 100 per condition).

**Metrics**:

*Position Bias*: For each condition, bias_first = r_first - 0.5, where r_first is the response rate for the first-listed word.
$$\text{avg\_pos\_bias} = \text{mean}(|\text{bias\_first}|)$$

*Semantic Bias*: For each word pair, we average the response rate across forward and reverse orderings to isolate word preference from position preference:
$$\text{avg\_bias} = \frac{r_{\text{word,fwd}} + r_{\text{word,rev}}}{2}$$
$$\text{red\_bias} = \max(0, |\text{avg\_bias} - 0.5| - 0.05)$$

The 0.05 tolerance accounts for sampling noise at N = 100.

**Scoring**:
$$\text{Score}_{\text{pos}} = \frac{1}{2} \times \max(0, 20 \times (1 - 2 \times \text{avg\_pos\_bias}))$$
$$\text{Score}_{\text{sem}} = \frac{1}{2} \times \max(0, 20 \times (1 - 2.22 \times \text{avg\_red\_bias}))$$
$$\text{Score}_6 = \text{Score}_{\text{pos}} + \text{Score}_{\text{sem}}$$

The coefficient 2.22 = 1/0.45 ensures the semantic score reaches 0 at the maximum possible reduced bias.

## Total Benchmark Score

$$\text{Total} = \text{Score}_1 + \text{Score}_2 + \text{Score}_3 + \text{Score}_4 + \text{Score}_5$$

Maximum possible score: 100 points.

## Benchmark Results

We evaluated 11 large language models using our benchmark (Table S1). Strikingly, no model achieved even 50% of the maximum possible score, with the best-performing model (Kimi K2) scoring only 46.6 out of 100. This confirms that probabilistic calibration failure is pervasive across all modern LLM architectures.

**Table S1: LLM Bias Benchmark Scores (out of 100)**

| Rank | Model | Total | Exp1 | Exp2 | Exp3 | Exp4 | Exp5 |
|------|-------|-------|------|------|------|------|------|
| 1 | Kimi K2 | 46.6 | 8.3 | 13.2 | 13.1 | 3.8 | 8.3 |
| 2 | Grok 4 | 42.5 | 2.7 | 9.8 | 12.9 | 5.6 | 11.4 |
| 3 | DeepSeek V3.2 | 40.6 | 1.9 | 12.7 | 11.9 | 3.8 | 10.3 |
| 4 | GPT-5.2 | 32.8 | 1.6 | 6.8 | 11.9 | 4.0 | 8.4 |
| 5 | GPT-5 Nano | 32.7 | 2.7 | 6.8 | 12.7 | 4.2 | 6.3 |
| 6 | Gemini 3 Pro | 32.0 | 0.7 | 6.4 | 11.6 | 4.4 | 8.8 |
| 7 | Claude 4.5 Sonnet | 31.0 | 1.2 | 3.5 | 8.0 | 0.0 | 18.3 |
| 8 | GPT-5 | 31.0 | 1.1 | 7.5 | 10.9 | 4.5 | 6.9 |
| 9 | Qwen3 VL 8B | 30.0 | 0.0 | 9.0 | 8.7 | 4.0 | 8.3 |
| 10 | Gemini 2.5 Pro | 28.6 | 0.4 | 7.8 | 10.9 | 3.8 | 5.8 |
| 11 | Claude Opus 4.5 | 22.2 | 0.0 | 2.7 | 9.6 | 4.2 | 5.7 |

For Experiment 1 (single binary flip), most models scored near zero, with 6 out of 11 models scoring below 2.0 out of 20. Only Kimi K2 showed meaningful deviation from perfect step-function behavior (S = 0.586, score = 8.3). This indicates that the step-function phenomenon observed in our main experiments is robust across model families.

In Experiment 2 (two sequential flips), we observed that models capable of self-correction on their second response achieved higher scores. Kimi K2 and DeepSeek V3.2 scored 13.2 and 12.7 respectively, suggesting that their ensemble averaging behavior partially compensates for individual response failures.

Experiment 3 (three-outcome distribution) showed relatively consistent performance across models, with scores ranging from 8.0 to 13.1. This suggests that multi-outcome probability failures are similarly challenging regardless of model architecture or training approach.

For Experiment 4 (real-world decision scenarios), all models performed poorly. Notably, Claude 4.5 Sonnet scored exactly 0.0, failing entirely on all game-theoretic prompts despite correctly reasoning through the optimal mixed strategy in its chain-of-thought outputs. This dissociation between correct reasoning and correct execution underscores that the failure to mix is not a reasoning failure but an output generation failure.

An unexpected finding emerged from Experiment 5 (semantic preference bias): Claude 4.5 Sonnet achieved the highest score of 18.3 out of 20, indicating minimal semantic and position bias. This result suggests that semantic neutrality and probabilistic calibration are independent capabilities—a model can exhibit strong word-choice neutrality while simultaneously failing at probability calibration.

## Implementation

The benchmark is implemented in Python and available at https://github.com/BiostateAIresearch/failure-to-mix. To run:

```bash
pip install aiohttp pandas numpy tqdm nest_asyncio
python benchmark.py --model "MODEL_NAME" --api-key "API_KEY"
```

Results are saved as CSV files (raw API responses) and JSON (summary scores). The benchmark uses the OpenRouter API for model access and supports parallel API calls with automatic retry logic.
