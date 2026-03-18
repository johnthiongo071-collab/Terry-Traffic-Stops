# Terry Stops — Predicting Arrest Outcomes with Machine Learning

## Overview

This project investigates patterns in police-reported Terry Stops in Seattle, using machine learning to predict stop outcomes — specifically, whether a stop results in an arrest. Inspired by the legal precedent established in *Terry v. Ohio*, which introduced the concept of "reasonable suspicion," this analysis explores how historically grounded policing data can — and cannot — be used responsibly in algorithmic decision-making contexts.

The work is aimed at stakeholders including **law enforcement agencies**, **government and policymakers**, **technology companies**, **civil rights organizations**, and the **general public** — all of whom have a stake in understanding how perceived demographics and contextual factors shape stop outcomes.

---

## Business and Data Understanding

### Stakeholder Audience

| Stakeholder | Interest |
|---|---|
| Law Enforcement Agencies | Improving consistency and efficiency in decision-making |
| Government & Policymakers | Regulating surveillance tools and ensuring legal alignment |
| Technology Companies | Building fair, accountable ML systems |
| Civil Rights Organizations | Evaluating potential bias in algorithmic outcomes |
| General Public | Understanding how automated decisions may affect them |

The core business objective is to **identify which variables most strongly predict stop outcomes** — particularly arrests — and to surface any patterns of bias embedded in historical policing data.

### Dataset

**Source:** Seattle Police Department Open Data Portal  
**Publisher:** City of Seattle  
**Update frequency:** Daily (last updated March 16, 2026)  
**Size:** 66,786 rows, each representing a unique Terry Stop  
**Scope:** Records span 2017 through early 2026, including officer-reported demographics and Computer-Aided Dispatch (CAD) system details.

### Target Variable

**Stop Resolution** — the final outcome of a stop (e.g., *Arrest*, *Field Contact*, *Offense Report*). This was binarized:
- `1` = Arrest
- `0` = All other outcomes

The dataset has a **class imbalance**: ~50,861 non-arrest stops vs. ~15,925 arrests (~3:1 ratio). This was addressed during modeling using SMOTE (see Modeling section).

### Features Used

| Category | Features |
|---|---|
| Subject Demographics | Age Group, Perceived Race, Perceived Gender |
| Officer Demographics | Race, Gender, Year of Birth |
| Context | Time of Day (bucketed from stop hour), Weapon Type (binary: present or not) |

> **Note on Ethics:** Because the dataset records "perceived" demographics as reported by officers, the analysis focuses on *how those perceptions influence outcomes* rather than treating them as objective ground truth. Features like Subject Perceived Race should be interpreted with caution — they reflect officer perception, not verified identity.

-**Metrics Justification:** In the context of the Terry Stops predictive model, we prioritize recall over precision to minimize the risk of "false negatives." ​In this specific scenario, failing to identify a stop that results in a significant outcome (like an arrest or a weapon recovery) is considered a more costly error than incorrectly flagging a stop that does not. By optimizing for recall, the model ensures that the majority of high-stakes interactions are captured for review, acknowledging that a higher rate of "false alarms" (lower precision) is a secondary concern compared to the potential safety or administrative oversight caused by missing an actual arrest.
---

## Modeling

Three models were built and compared:

**Model 1 — Baseline Logistic Regression**  
A standard logistic regression with no class imbalance handling. Used as a reference point.

**Model 2 — Tuned Logistic Regression (SMOTE)**  
To address the ~3:1 class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data, synthetically generating additional arrest examples until both classes were balanced. The logistic regression was then trained on this resampled dataset. This improves recall for the minority class (arrests) compared to the baseline.

**Model 3 — Decision Tree (Balanced, Entropy)**  
A Decision Tree classifier with `max_depth=5`, `class_weight='balanced'`, and entropy criterion. Provides interpretable splits and explicit feature importances.

### Feature Engineering

Stop timestamps were converted into **time-of-day buckets** rather than raw hour values (0–23):

| Bucket | Hours |
|---|---|
| Morning | 05:00 – 11:59 |
| Afternoon | 12:00 – 16:59 |
| Evening | 17:00 – 20:59 |
| Late Night | 21:00 – 04:59 |

This avoids treating hour as a continuous number (which implies hour 23 is "close to" hour 22 but "far from" hour 1, even though midnight stops share behavioral patterns across 23:00 and 01:00).

All models were trained on 70% of the data and evaluated on a 30% held-out test set. Preprocessing included:
- **One-Hot Encoding** for all categorical features (age group, officer/subject demographics, time of day)
- **Standard Scaling** for numerical features (Officer YOB)

---

## Evaluation

### Classification Performance (Test Set)

| Metric | Logistic Regression (SMOTE) | Decision Tree (Balanced) |
|---|---|---|
| Overall Accuracy | 56% | 50% |
| Arrest Recall | 60% | 67% |
| Non-Arrest Recall | 55% | 45% |
| Arrest Precision | 29% | 27% |
| Arrest F1 | 0.39 | 0.39 |

> Both models achieve similar F1 scores for arrests (~0.39). The Decision Tree captures more true arrests (higher recall) but at the cost of more false positives. For a policing context where **missing a true arrest is the higher-stakes error**, the Decision Tree's higher recall may be preferable — but this trade-off requires careful domain judgment.

### ROC-AUC Scores

AUC scores are read from the ROC curve plot in the notebook. Both models outperform random chance (AUC = 0.50), but neither achieves strong discriminative power.

### Top Predictive Features (Decision Tree)

Feature importances from the decision tree show that **time of day** and **officer year of birth** are among the strongest predictors, followed by perceived race categories. This raises important questions about how structural factors (when stops occur, which officers conduct them) interact with outcomes — and whether perceived race is serving as a proxy for those structural factors.

---

## Conclusion

Both models show **limited predictive power**. This suggests that the variables available — including perceived demographics — do not strongly or reliably predict arrest outcomes on their own. The fact that **perceived race** still appears among the top decision tree features is significant and warrants further fairness analysis.

Key takeaways:

- Class imbalance (~3:1) was a significant challenge; SMOTE improved arrest recall from baseline but overall accuracy remains modest
- Converting stop hour to time-of-day buckets (Morning / Afternoon / Evening / Late Night) produced a more meaningful temporal feature than raw hour values
- The modest performance of these models suggests arrest decisions are complex and not easily reducible to the recorded features — or that important factors are absent from the dataset
- **Perceived race appearing as a top feature is a red flag**, not a justification for its use — it likely reflects historical policing patterns rather than causal factors
- Any deployment of ML in policing contexts must grapple seriously with fairness, transparency, and the risk of amplifying historical bias at scale

---

## Limitations

- **Perceived demographics are not objective:** All demographic features are officer-reported perceptions, introducing a layer of subjectivity and potential bias into every model trained on this data
- **These models should not be used for real-world decision-making.** Performance is too low, and the features carry significant fairness risks
- **Dropped features may matter:** Precinct, sector, beat, and call type were excluded from modeling. Location and call context could be meaningful predictors and deserve exploration in future iterations
- **No cross-validation was used:** A single 70/30 train-test split was used; results may vary with different random seeds. Cross-validation would provide more reliable performance estimates
- **AUC scores are approximate:** The ROC curve is plotted from the notebook but exact AUC values should be read directly from the chart output

---

## Recommendations for Future Work

- **Run a fairness audit:** Calculate precision and recall broken down by Subject Perceived Race to surface whether model performance is uneven across demographic groups
- **Try stronger models:** Random Forest or XGBoost typically outperform single decision trees and logistic regression on tabular classification tasks
- **Use cross-validation:** Replace the single train/test split with k-fold cross-validation for more robust evaluation
- **Re-introduce location features:** Precinct, sector, and beat were dropped early — these may carry genuine predictive signal worth exploring
- **Add a calibration curve:** Check whether predicted probabilities reflect real-world likelihoods, especially important if probability scores are ever used downstream
- **Frame evaluation around recall:** For an arrest prediction task, missing a true arrest is the higher-cost error — optimize and report metrics accordingly

---

## Reproducibility

**Libraries required:**
```
pandas
numpy
scikit-learn
imbalanced-learn   # for SMOTE
matplotlib
seaborn
```

Install with:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

**Data:** `Terry_Stops.csv` — available from the [City of Seattle Open Data Portal](https://data.seattle.gov/)

Run all cells in order from top to bottom in `traffic.ipynb`.
