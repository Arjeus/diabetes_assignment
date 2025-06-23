**Summary Report – Diabetes 30-Day Readmission Notebook**

---

### 1  Approach & rationale

| Phase                        | Main actions                                                                                                                                                                                                                                                                       | Why it was done                                                                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data cleaning**            | • Removed invalid gender records and the *weight* field (≃97 % missing) • Recast ID columns to *category* • Replaced “?” with “Unknown” and filtered hospice/death discharges                                                                                                      | Eliminate noise and non-informative attributes, keep categorical memory-efficient, exclude cases that do not lead to readmission management decisions.              |
| **Feature engineering**      | • Bucketed ICD-9 codes into clinically meaningful groups while keeping diabetes-related “250.xxx” explicit                                   | Preserve dose-direction signal; curb high-cardinality sparse variables; retain pathology detail central to readmission risk.                                        |
| **Handling class imbalance** | • Observed positive rate 11.3 % • In the baseline experiments performed random up-sampling; in the final pipeline inserted **SMOTE** before the classifier                                                                                                                         | Avoid majority-class bias and allow the learner to “see” enough readmission examples.                                                                               |
| **Modeling strategy**        | 1. Baselines: Logistic Regression, Random Forest, XGBoost (default 100 trees)  2. Hyper-parameter optimisation of XGBoost with **Optuna** (100 TPE trials) inside a SMOTE pipeline  3. Deep-learning alternative with **PyTorch-Tabular**, likewise tuned with Optuna (50 trials)  | XGBoost is the de-facto strong learner for tabular data; RF & LR give interpretable baselines; PT-Tabular gauges whether modern DL architectures can close the gap. |
| **Interpretability**         | • Permutation importance for both learners  • Global SHAP analysis on the tuned XGBoost model (5 000 test rows)                                                                                                                                                                    | Identify actionable factors and verify that the model does not rely on spurious artefacts.                                                                          |

---

### 2  Key results & interpretations

**2.1 Baseline comparison (no tuning)**

| Model               | Accuracy | Precision | Recall    | F1        | ROC-AUC    |
| ------------------- | -------- | --------- | --------- | --------- | ---------- |
| Logistic Reg.       | 0.718    | 0.221     | 0.590 | 0.322     | 0.723      |
| Random Forest       | 0.640    | 0.204     | **0.752** | 0.322     | 0.744      |
| XGBoost (100 trees) | 0.704    | **0.237** | 0.729     | **0.358** | **0.782**  |

*Observation:* Tree-based methods already outperform logistic regression on F1 and AUC, confirming the non-linear nature of the task.

---

**2.2 Optimised models**

| Metric    | Tuned XGBoost | Tuned PyTorch-Tabular |
| --------- | ------------- | --------------------- |
| Accuracy  | **0.887**     | 0.747                 |
| Precision | **0.505**     | 0.232                 |
| Recall    | **0.054**         | 0.532             |
| F1        | 0.098         | **0.323**             |
| ROC-AUC   | **0.786**     | 0.653                 |

*Interpretation*

* **XGBoost** excels at overall discrimination (AUC) and precision but misses many true readmissions (low recall). Confusion matrix shows 185 true positives versus 3 222 false negatives .
* **PyTorch-Tabular** trades accuracy for recall—capturing 53 % of readmissions—potentially preferable in safety-critical screening, albeit with more false alarms .

---

**2.3 Feature importance (SHAP – tuned XGBoost)**

| Rank | Feature (description)                                         | SHAP mean | Insight                                           |
| ---- | ------------------------------------------------------------- | --------- | ------------------------------------------------- |
| 1    | `patient_freq` (historical admission proportion)              | **1.05**  | Frequent prior admissions → chronic instability   |
| 2-4  | `age` bands 70-80, 60-70, 80-90                               | 0.32-0.21 | Older age markedly elevates risk                  |
| 5    | `number_inpatient`                                            | 0.26      | Recent in-hospital stays signal complications     |
| 6    | `time_in_hospital`                                            | 0.24      | Longer index admission hints severity             |
| 7    | `race_Caucasian`                                              | 0.19      | Proxy for dataset composition (≈75 % Caucasian)   |
| 8    | `disch_reduced_3` (home-health-care discharge)                | 0.17      | Early send-home with assistance may imply frailty |
| 9-15 | Diabetes ICD-250 codes, *insulin steady/up*, procedures count | 0.15-0.11 | Direct clinical burden and therapeutic intensity  |

The SHAP bar plot and beeswarm confirm that no single laboratory value dominates; rather, utilisation history and age drive predictions.

---

### 3  Recommendations & further analysis

1. **Improve recall while retaining precision for XGBoost**

   * Use class-weighted or focal-loss variants, or optimise the decision threshold for F<sub>β</sub> (β > 1).
   * Explore calibrated probability cut-offs or *precision-recall* curve operating points.

2. **Refine imbalance handling**

   * Apply class weighting to guarantee higher recalls.
   * Investigate *cost-sensitive boosting* (e.g., XGBoost’s `scale_pos_weight`).

4. **Fairness & external validity**

   * The model’s race signal likely reflects sampling bias; perform subgroup performance audits and, if necessary, re-weight or debias.
   * Validate on a hospital outside the original 130-institution cohort.

5. **Ensemble or hybrid strategy**

   * Combine high-precision XGBoost with high-recall PT-Tabular in a two-stage cascade: flag by recall-oriented model, confirm with precision model.

6. **Interpretability**

   * Try SHAP on deep learning models.
   * Study feature importance on trained deep learning models: improve training so that the model responds to important features.

