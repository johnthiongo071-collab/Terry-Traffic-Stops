# Terry Stops — Predictive Modeling of Stop Outcomes

## Business Understanding

In the modern digital era (circa 2026), the rapid expansion of machine learning has significantly influenced how surveillance is conducted. This project is inspired by the legal precedent established in *Terry v. Ohio*, which introduced the concept of "reasonable suspicion" in stop-and-frisk practices. While originally rooted in human judgment, similar decision-making processes are now being translated into algorithmic systems. This raises important questions about how historical data, when used to train machine learning models, may shape modern surveillance outcomes.

**Business Objective:** Explore the impact of using historically grounded policing datasets to train machine learning models in a digital surveillance context.

**Key Stakeholders:**
- Law Enforcement Agencies
- Government & Policymakers
- Technology Companies
- Civil Rights Organizations
- General Public

**Business Problem:**
- **Pattern Identification:** Analyzing variables used to justify "reasonable suspicion" decisions.
- **Predictive Modeling:** Training classifiers to predict stop outcomes based on observed variables like officer gender, subject perceived race, and stop location.

---

## Data Understanding

The dataset is the **Terry Stops dataset**, provided by the City of Seattle via the Seattle Police Department (SPD) Open Data portal. It contains records of police-reported stops under the legal precedent of *Terry v. Ohio*.

- **Source:** Seattle Police Department
- **Rows:** 66,800+ (each representing a unique stop)
- **Scope:** Records from 2017 through early 2026
- **Update Frequency:** Daily (last updated March 16, 2026)

**Target Variable:** `Stop Resolution` — binarized as `1 = Arrest`, `0 = No Arrest`

**Key Predictor Variables:**
- Subject Demographics: Age Group, Perceived Race, Perceived Gender
- Officer Demographics: Race, Gender, Year of Birth
- Contextual Data: Precinct, Sector, Beat, Occurred Date
- Stop Specifics: Weapon Type, Arrest Flag, Frisk Flag, Initial Call Type

> **Note on Data Ethics:** Because this data relies on "perceived" demographics reported by officers, the analysis focuses on the impact of these perceptions and potential biases within the recorded data rather than an objective demographic census.

---

## Data Preparation

Key preprocessing steps:

1. Dropped non-informative or redundant columns (e.g., IDs, raw date strings, administrative columns).
2. Created a **binary target** column: `1` if `Stop Resolution == 'Arrest'`, `0` otherwise.
3. Created a **`weapon_binary`** feature: `1` if a weapon was present, `0` otherwise.
4. Fixed a data entry error in `Subject Age Group` (`'17-Jan'` → `'1 - 17'`).
5. Filled remaining NaN values with `'Unknown'` for categorical columns.
6. Extracted **Time of Day** buckets (Morning, Afternoon, Evening, Late Night) from `Occurred Date`.
7. Applied **OneHotEncoding** to categorical features and **StandardScaler** to numerical features via a `ColumnTransformer`.

---

## Univariate and Bivariate Analysis

Before modelling, three exploratory plots were produced to understand the distribution of key variables and their relationship with the target outcome.

### 1. Stop Outcomes by Subject Race
A count plot comparing arrest vs. non-arrest outcomes broken down by the subject's perceived race. This chart reveals whether certain racial groups are disproportionately represented among stops that end in arrest — a central question for assessing bias in the dataset. Across all racial groups, non-arrest outcomes dominate, but the relative arrest rate varies, making perceived race a potentially significant predictor.

### 2. Weapon Presence by Time of Day
A count plot showing how often a weapon was present during stops across four time-of-day buckets (Morning, Afternoon, Evening, Late Night). This bivariate view explores whether weapon-involved stops cluster at particular times — for example, late-night stops may carry a higher rate of weapon presence, which could interact with both arrest likelihood and officer decision-making.

### 3. Distribution of Officer Year of Birth
A histogram with a KDE curve showing the spread of officer year of birth (restricted to 1950–2002, reflecting the active workforce). This univariate chart gives a sense of the age profile of officers in the dataset — a mostly younger-to-mid-career workforce — and flags whether officer experience (proxied by age) may be worth investigating as a factor in stop outcomes.

---

## Modelling

### Model 1: Baseline Logistic Regression
A standard logistic regression trained on the transformed training data without any class imbalance correction.

### Model 2: Tuned Logistic Regression (SMOTE)
The target class is imbalanced — approximately **76% No Arrest** vs. **24% Arrest**. To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training data before fitting the logistic regression.

### Model 3: Decision Tree (SMOTE)
A Decision Tree Classifier (`max_depth=5`, `class_weight='balanced'`, `criterion='entropy'`) trained on the SMOTE-resampled data.

---

## Evaluation

### Why Recall Over Precision?

In this context, **recall is the primary evaluation metric** rather than precision. This is a deliberate, domain-informed choice:

- A **false negative** (predicting "No Arrest" when an arrest actually occurs) means the model fails to flag a high-risk stop — the more serious error in a surveillance and policing context, as it could mean patterns of harmful stops go undetected.
- A **false positive** (predicting "Arrest" when no arrest occurs) is less costly — it flags a stop for review that turns out to be routine.

In other words, it is more acceptable for the model to over-predict arrests than to systematically miss them. Precision measures how many predicted arrests were real; recall measures how many real arrests the model actually caught. For this problem, catching real arrests matters more than avoiding false alarms.

### Classification Reports

| Metric | Logistic Regression (Tuned) | Decision Tree |
|---|---|---|
| Accuracy | 0.56 | 0.45 |
| Precision (Arrest) | 0.29 | 0.26 |
| Recall (Arrest) | 0.59 | 0.73 |
| F1-Score (Arrest) | 0.39 | 0.39 |

### Recommended Model: Decision Tree

The **Decision Tree is the preferred model** for this task. Although its overall accuracy (0.45) is lower than the Logistic Regression (0.56), accuracy is a misleading metric on an imbalanced dataset — a model that simply predicted "No Arrest" every time would score ~76% accuracy while being completely useless.

On the metric that matters most — **recall for the Arrest class** — the Decision Tree significantly outperforms Logistic Regression (**0.73 vs. 0.59**), correctly identifying nearly three-quarters of all actual arrests. The AUC scores are comparable (0.62 vs. 0.63), confirming that both models have similar overall discriminatory power, but the Decision Tree converts that power into better detection of the minority class. Cross-validation further supports this, with a mean recall of 0.75 across folds.

### AUC Scores

The following table summarises the **Area Under the ROC Curve (AUC)** for each model evaluated on the held-out test set. A higher AUC indicates better ability to distinguish between arrest and non-arrest outcomes across all classification thresholds.

| Model | AUC Score |
|---|---|
| Logistic Regression (Tuned) | 0.63 |
| Decision Tree | 0.62 |
| Random Guess (Baseline) | 0.50 |

> Both models outperform the random baseline. The AUC scores are nearly identical, meaning the models have comparable overall discriminatory power — the key differentiator is how that power translates into recall on the Arrest class, where the Decision Tree has a clear advantage.

### Cross-Validation (Decision Tree)

To assess the stability of the Decision Tree model, **5-fold cross-validation** was performed on the SMOTE-resampled training data using **Recall** as the scoring metric (prioritising the correct identification of arrest outcomes).

| Fold | Recall Score |
|---|---|
| 1 | 0.633 |
| 2 | 0.755 |
| 3 | 0.779 |
| 4 | 0.781 |
| 5 | 0.784 |
| **Mean** | **0.75** |
| **Std Dev** | **0.06** |

> The mean recall of **0.75** indicates the model correctly identifies roughly 3 in 4 arrest cases on average. The standard deviation of **0.06** suggests moderate variability across folds, with Fold 1 notably lower — this may reflect temporal patterns in the data given the train/test split.

### Top 10 Features (Decision Tree)

Feature importance analysis from the Decision Tree revealed that a small set of variables drives most of the predictive power for arrest outcomes. The top features are visualised in the notebook and centre primarily on **weapon presence** and **subject/officer demographic variables**.

---

## Key Takeaways

- Class imbalance significantly impacts model performance; SMOTE helps boost recall for the minority class (Arrest).
- Both models achieve moderate discriminatory ability (AUC ~0.62–0.63), suggesting that the available features have limited predictive power for arrest outcomes.
- Weapon presence and perceived subject demographics are among the strongest predictors of arrest — a finding with important ethical implications regarding algorithmic bias in policing systems.
- Cross-validation confirms that the Decision Tree model is reasonably stable, though there is fold-to-fold variation that warrants further investigation.
