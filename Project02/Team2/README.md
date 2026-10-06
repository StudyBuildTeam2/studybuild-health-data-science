# Fair, Interpretable, and Actionable Readmission Risk Modeling
## Diabetes 130-US Hospitals (1999–2008)

**StudyBuild — Project 02 | Best-practice pipeline aligned with 2024–2025 papers**

---

### Clinical Problem
Predict 30-day early readmission (`<30`) among diabetic inpatients and examine fairness across race, gender and age. This is an educational risk-stratification exercise, **not** a deployed clinical system.

---

### What Recent Strong Papers Do (2024–2025)
On this exact dataset the honest state-of-the-art is:

| Model              | Typical AUC-ROC | F1 (`<30`) | Notes                          |
|--------------------|-----------------|------------|--------------------------------|
| **XGBoost / LightGBM / CatBoost** | **0.66 – 0.70** | ≈ 0.27–0.29 | Best and most used             |
| Random Forest      | 0.63 – 0.68     | ≈ 0.25     | Strong baseline                |
| Logistic Regression| 0.64 – 0.66     | ≈ 0.25     | Fully interpretable            |

High Accuracy numbers (0.88–0.89) almost always hide very low Recall on the minority class and are **not** clinically useful.

Our final pipeline reproduces these realistic numbers:
- **LightGBM**: AUC ≈ **0.684**, F1 ≈ **0.288**, Recall ≈ **0.47**
- **XGBoost**: AUC ≈ **0.682**, F1 ≈ **0.287**, Recall ≈ **0.45**

---

### Recommended Final Pipeline (`src/final_best_pipeline.py`)

1. ICD-9 diagnosis grouping into clinically meaningful categories  
2. Feature engineering (`total_prior_visits`, `meds_per_day`, `lab_per_day`)  
3. Drop high-missing columns (weight, A1C, max_glu_serum, specialty, payer)  
4. Label-encode remaining categoricals  
5. Stratified train/test split + fixed random seed  
6. **XGBoost + LightGBM** with moderate `scale_pos_weight`  
7. Threshold tuning on a validation split to maximise F1 of the `<30` class  
8. Report AUC-ROC, PR-AUC, Recall, F1 and Accuracy together  
9. Feature importance (gain) for interpretation  

---

### How to Run

```bash
pip install -r requirements.txt
# make sure xgboost and lightgbm are installed

# Best modern pipeline (recommended)
python src/final_best_pipeline.py

# Full exploratory notebook that answers Q1–Q9
jupyter notebook notebooks/analysis.ipynb
```

---

### Project Structure
```
readmission-risk-modeling/
├── README.md
├── requirements.txt
├── data/
│   ├── diabetic_data.csv
│   └── IDS_mapping.csv
├── notebooks/analysis.ipynb          # answers all 9 questions
├── src/final_best_pipeline.py        # ← recommended final code
├── figures/
├── report/summary.md
└── individual-submissions/
```

---

### Mapping to Project Questions
| Phase                     | Questions          |
|---------------------------|--------------------|
| Data preparation & EDA    | Q1, Q2, Q3         |
| Modeling & Evaluation     | Q4, Q5, Q6, Q7, Q8 |
| Clinical report           | Q9                 |

---

### Key Scientific Messages (use in your report / video)
- Accuracy alone is misleading because of severe class imbalance (~11% early readmissions).  
- We prioritise **Recall and F1 of the `<30` class**.  
- Observed associations are **not** causal.  
- Performance differences across race / gender / age must be reported honestly.  
- Data limitations: 1999–2008, ICD-9 only, high missingness, no external validation.

---

### Citation
Strack et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates.  
Dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008  
DOI: 10.24432/C5230J

**StudyBuild — Learn • Build • Apply**
