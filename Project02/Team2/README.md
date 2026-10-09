# Fair, Interpretable, and Actionable Readmission Risk Modeling
## Diabetes 130-US Hospitals (1999–2008)

**StudyBuild — Project 02 | Digital Health & Medical Data Science Track**

---

### Clinical Problem
Predict 30-day early hospital readmission (`<30`) among diabetic inpatients and check whether the model behaves fairly across race, gender, and age groups.

> This is an **educational** risk-stratification exercise, not a deployed clinical decision system.

---

### Final Models & Results (aligned with 2024–2025 papers)

| Model       | AUC-ROC | F1 (`<30`) | Recall (`<30`) |
|-------------|---------|------------|----------------|
| **LightGBM**| **0.684** | **0.288** | **0.47** |
| **XGBoost** | **0.682** | **0.287** | **0.45** |

These numbers match honest recent literature (typical AUC 0.66–0.70).

---

### Project Structure
```
readmission-risk-modeling/
├── README.md
├── requirements.txt
├── data/
│   ├── diabetic_data.csv
│   ├── IDS_mapping.csv
│   └── README.md
├── notebooks/
│   └── analysis.ipynb          # Full EDA + Q1–Q9
├── src/
│   └── final_best_pipeline.py  # Recommended final code (XGBoost + LightGBM)
├── figures/                    # All plots
├── report/
│   └── summary.md
└── individual-submissions/
```

---

### How to Run
```bash
pip install -r requirements.txt
python src/final_best_pipeline.py
# or
jupyter notebook notebooks/analysis.ipynb
```

---

### Mapping to Project Questions
| Phase | Questions |
|-------|-----------|
| Data preparation & EDA | Q1, Q2, Q3 |
| Modeling & Evaluation | Q4, Q5, Q6, Q7, Q8 |
| Clinical interpretation | Q9 |

---

### Key Design Choices
- ICD-9 grouped into clinical categories
- Feature engineering: `total_prior_visits`, `meds_per_day`, `lab_per_day`
- Dropped high-missing / ID / constant columns
- Stratified split + fixed seed (42)
- Class imbalance handled with `scale_pos_weight`
- Threshold tuned on validation for best F1 of `<30`
- Fairness checked by race, gender, age

### Top Features (XGBoost)
1. number_inpatient  
2. discharge_disposition_id  
3. total_prior_visits  
4. diabetesMed  
5. diag1_cat  

---

### Citation
Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates.  
Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
DOI: 10.24432/C5230J

**StudyBuild — Learn • Build • Apply**
