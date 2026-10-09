# Clinical & Fairness Summary

## Main patterns
- Early readmission (<30 days) ≈ 11.2% of encounters.
- Strongest signal: prior inpatient visits.
- Engineered features (total_prior_visits, meds_per_day) were useful.

## Model performance
- LightGBM / XGBoost: AUC ≈ 0.68, F1(<30) ≈ 0.29, Recall ≈ 0.45–0.47.
- Accuracy alone is misleading due to class imbalance.

## Fairness
- Recall differs across race, gender, and age brackets.
- Differences are reported transparently.

## Limitations
- Data period 1999–2008, ICD-9 only, high missingness, no external validation.
- Encounter-level split (not purely patient-level).

## Recommendation
Use only as a risk-stratification aid for discharge planning — never for automated care denial.
