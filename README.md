# Instructor Effectiveness Modeling

**EdTech analytics project** — building a defensible, well-justified pipeline to score and classify instructor effectiveness from batch-level performance data.

## Objective

1. Explore batch-level instructor performance data (EDA)
2. Define a composite **Instructor Effectiveness Score (IES)**
3. Aggregate batch-level records to the instructor level, correcting for sample-size bias
4. Train and evaluate a classifier that predicts an instructor's effectiveness tier (Low / Medium / High)
5. Interpret the results for non-technical EdTech stakeholders

## Dataset

2,000 batch-level records covering **120 unique instructors** and **25 courses**. Each row is one course batch taught by one instructor, with nine performance metrics: `completion_rate`, `dropout_rate`, `avg_score_improvement`, `avg_quiz_score`, `avg_watch_time`, `assignment_submission_rate`, `forum_activity_rate`, `avg_feedback_score`, `feedback_response_rate`.

## Key Findings (EDA)

- No missing values across any column.
- Instructors teach unevenly-sized samples — between **7 and 31 batches** each (median 17). Low-batch instructors have noisier average metrics from sampling variance alone, not necessarily different skill.
- The nine metrics sit on incompatible scales (0–1 rates, 1–5 ratings, 0–40 point improvements) — any composite score must normalize before combining.
- `completion_rate` and `dropout_rate` are almost perfectly redundant (**correlation ≈ −0.95**). Including both at full weight would double-count one signal.

## Methodology

### 1. Composite Score
Nine raw metrics are grouped into three weighted pillars — **Learner Outcomes (50%)**, **Engagement (30%)**, **Feedback/Satisfaction (20%)** — and combined into a single 0–100 **Instructor Effectiveness Score** using min-max normalized, weighted averages. `dropout_rate` is deliberately excluded from the formula itself since it's mathematically ≈ (1 − `completion_rate`).

### 2. Empirical-Bayes Shrinkage
To prevent low-batch-count instructors from being over-represented at both score extremes due to noise, each instructor's mean is pulled toward the global mean, weighted inversely by their batch count:

```
x̂ = (n / (n+k)) * instructor_mean + (k / (n+k)) * global_mean
```

with `k = 10`. Instructors near the median batch count are shrunk about halfway; instructors with very few batches are shrunk substantially; high-volume instructors are barely touched.

### 3. Classification
Instructors are tiered into Low/Medium/High via tertiles of the Effectiveness Score, then a **Random Forest** and **Logistic Regression** classifier are trained to predict tier from the underlying mean/std features — explicitly *excluding* the Effectiveness Score itself to avoid direct target leakage.

- Evaluated with a held-out stratified test split **and** 5-fold stratified cross-validation (given the small N of 120 instructors).
- Cross-validated **macro-F1 ≈ 0.83–0.90**.
- `dropout_rate_mean` shows high feature importance despite being excluded from the scoring formula — a concrete, diagnosed example of a confounded/redundant variable rather than new signal, since it mirrors `completion_rate`.

## Key Takeaways for Stakeholders

- Course completion and learning-gain metrics are the strongest, most trustworthy drivers of effectiveness — both by design and confirmed by the model.
- Feedback-based metrics carry real risk of self-selection and halo-effect bias and are weighted accordingly.
- The model is a **coaching diagnostic, not a standalone evaluation tool** — it cannot separate instructor skill from confounds like course difficulty, non-random course assignment, and learner composition, and has not been validated against independent, downstream outcomes.

## Tech Stack

`Python` · `Pandas` · `NumPy` · `scikit-learn` (Random Forest, Logistic Regression, StratifiedKFold) · `Matplotlib` · `Seaborn`

## Repository Structure

```
├── Instructor_Effectiveness_Modeling.ipynb   # Full analysis notebook
└── README.md
```

## Limitations

This analysis is intentionally transparent about its own boundaries: the effectiveness tiers are derived from self-chosen formula weights (not validated against independent outcomes), the sample size (120 instructors) is small, there is no causal or longitudinal grounding, and no fairness/bias audit has been performed. See the notebook's "Mandatory Analysis Questions" section for a full discussion of failure modes and recommended additional data.
