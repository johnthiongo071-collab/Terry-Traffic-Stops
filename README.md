# Terry Stops — Predicting Arrest Outcomes with Machine Learning

## Overview

This project investigates patterns in police-reported Terry Stops in Seattle, using machine learning to predict stop outcomes (specifically, whether a stop results in an arrest). Inspired by the legal precedent established in *Terry v. Ohio*, which introduced the concept of "reasonable suspicion," this analysis explores how historically grounded policing data can — and cannot — be used responsibly in algorithmic decision-making contexts.

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
**Size:** 66,800+ rows, each representing a unique Terry Stop  
**Scope:** Records span 2017 through early 2026, including officer-reported demographics and Computer-Aided Dispatch (CAD) system details.

### Target Variable

**Stop Resolution** — the final outcome of a stop (e.g., *Arrest*, *Field Contact*, *Offense Report*). This was binarized:
- `1` = Arrest  
- `0` = All other outcomes

The dataset has a **class imbalance**: ~50,861 non-arrest stops vs. ~15,925 arrests.

### Features Used

| Category | Features |
|---|---|
| Subject Demographics | Age Group, Perceived Race, Perceived Gender |
| Officer Demographics | Race, Gender, Year of Birth |
| Context | Stop Hour (extracted from date), Weapon Type (binary: present or not) |

> **Note on Ethics:** Because the dataset records "perceived" demographics as reported by officers, the analysis focuses on *how those perceptions influence outcomes* rather than treating them as objective ground truth.

---

## Modeling

Three models were built and compared:

**Model 1 — Baseline Logistic Regression**  
A standard logistic regression with no class imbalance handling. Used as a reference point.

**Model 2 — Tuned Logistic Regression (Balanced Weights)**  
After observing significant class imbalance (~3:1 ratio of non-arrests to arrests), this model used `class_weight='balanced'` to give the minority class (arrests) proportionally more weight during training. This improves recall for arrests at the cost of some overall accuracy.

**Model 3 — Decision Tree (Balanced, Entropy)**  
A Decision Tree classifier with `max_depth=5`, `class_weight='balanced'`, and entropy criterion. Provides interpretable splits and explicit feature importances.

All models were trained on 70% of the data and evaluated on a 30% held-out test set. Preprocessing included:
- **One-Hot Encoding** for categorical features (age group, officer/subject demographics)
- **Standard Scaling** for numerical features (Officer YOB, Stop Hour)

---

## Evaluation

### Classification Performance (Test Set)

| Metric | Logistic Regression (Balanced) | Decision Tree (Balanced) |
|---|---|---|
| Overall Accuracy | 57% | 49% |
| Arrest Recall | 57% | 69% |
| Non-Arrest Recall | 57% | 42% |
| Arrest Precision | 29% | 27% |
| Arrest F1 | 0.38 | 0.39 |

### ROC-AUC Scores

| Model | AUC |
|---|---|
| Logistic Regression (Balanced) | 0.63 |
| Decision Tree (Balanced) | 0.59 |

The Logistic Regression model achieves marginally better discrimination (AUC 0.63 vs. 0.59), while the Decision Tree captures more true arrests (higher recall at 69%). Both outperform a random baseline (AUC = 0.50), but neither achieves strong predictive performance — which is itself a meaningful finding.

### Top Predictive Features (Decision Tree)

The decision tree's feature importances reveal that **stop hour** and **officer year of birth** are the strongest predictors of arrest, followed by perceived race categories. This raises important questions about how time-of-day patterns and individual officer characteristics interact with outcomes.

---

## Conclusion

Both models show **limited predictive power**, with AUC scores in the 0.59–0.63 range. This suggests that the variables available in this dataset — including perceived demographics — do not strongly or reliably predict arrest outcomes. However, the fact that **perceived race** still appears among the top features is noteworthy and warrants further investigation.

Key takeaways:

- Predicting arrests from demographic and contextual stop data is inherently limited and fraught with bias risk
- Class imbalance required explicit handling; balanced models improved arrest recall but reduced overall accuracy
- The modest performance of these models suggests that stop outcomes are complex and not easily reducible to the recorded features — or that arrest decisions involve factors not captured in this dataset
- Any deployment of ML in policing contexts must grapple seriously with fairness, transparency, and the potential for historical bias to be amplified at scale

This project demonstrates both the *possibility* of building predictive models from policing data and the *limits and risks* of doing so without careful ethical framing.
