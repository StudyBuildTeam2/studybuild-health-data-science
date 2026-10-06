# Clinical Analysis & Fairness Summary
## Fair, Interpretable, and Actionable Readmission Risk Modeling

### 1. Main Patterns (from Q1–Q3)
- Early readmission (<30 days) occurs in approximately **11.2%** of encounters.
- Patients who are readmitted early tend to have:
  - Longer hospital stays
  - Higher number of medications
  - Significantly more prior inpatient visits
  - Higher number of diagnoses
- Prior utilization history is the strongest observable signal in the dataset.
- Observed associations must **not** be interpreted as causal.

### 2. Model Performance (Q4–Q5)
- Baseline models (Logistic Regression + Decision Tree) provide a transparent risk signal.
- Overall accuracy is **not** a sufficient metric due to class imbalance.
- Primary evaluation focuses on **Recall** and **F1-score of the <30 class**.
- Missing an early-readmission case (False Negative) has higher clinical cost than a False Positive in this context.

### 3. Error Analysis (Q6)
- False Negatives tend to occur among patients with less extreme utilization numbers (harder borderline cases).
- Errors are not randomly distributed; they show some concentration by age and prior utilization.

### 4. Feature Drivers (Q7)
- Strongest contributors (Logistic Regression coefficients & permutation importance):
  - `number_inpatient` (prior hospitalizations)
  - Length of stay and medication count
  - Certain discharge dispositions and admission types
  - Selected diagnosis categories
- These are associative, not causal, findings.

### 5. Fairness Findings (Q8)
- Model Recall and False-Negative Rate are **not identical** across race, gender, and age brackets.
- Differences are reported transparently.
- Possible explanations: sample-size imbalance, differing base rates, unmeasured social determinants.
- The model should **not** be used to make decisions that systematically disadvantage any subgroup without further mitigation and prospective monitoring.

### 6. Recommendations for Hospital Leadership (Q9)
1. Treat the model strictly as a **risk-stratification aid** for enhanced discharge planning and early follow-up — never for care denial.
2. Prioritize patients with high prior inpatient utilization for medication reconciliation and rapid outpatient contact.
3. Before any real-world use:
   - Retrain on contemporary data (post-2010, ICD-10 era)
   - Add social-determinant and medication-adherence variables
   - Perform prospective validation
   - Implement continuous fairness monitoring
4. Improve structured capture of A1C, weight, and specialty information going forward.

### 7. Critical Limitations
- Data period: 1999–2008
- ICD-9 coding only
- High missingness in clinically important variables (weight, A1C, glucose)
- Encounter-level rather than purely patient-level split
- No external validation set
- US hospital setting only

**This analysis is educational. It is not a clinical decision-support tool.**
