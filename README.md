# Santander Customer Satisfaction Prediction

This project explores customer satisfaction prediction using the **Santander Bank dataset** (Kaggle competition). The goal was to identify dissatisfied customers and uncover key drivers of dissatisfaction through machine learning, dimensionality reduction, and model interpretability techniques.  

---

## 📌 Problem Statement
Santander Bank provided anonymized customer data with **370+ features** and a highly imbalanced target variable (`TARGET = 1` → dissatisfied customer).  
The objective was to build a classification model that:
- Accurately predicts dissatisfied customers.  
- Handles class imbalance.  
- Provides interpretable insights into key satisfaction drivers.  

---

## 📊 Dataset
- **Source:** [Kaggle – Santander Customer Satisfaction](https://www.kaggle.com/c/santander-customer-satisfaction)  
- **Training data:** 76,020 rows × 371 columns (including `TARGET`)  
- **Test data:** 75,817 rows × 370 columns  
- Features are anonymized for privacy/security reasons.  
- Only ~4% of customers are in the dissatisfied class (severe imbalance).  

---

## ⚙️ Methodology
1. **Data Preprocessing & EDA**  
   - Missing value checks, outlier detection, scaling.  
   - Dimensionality reduction via **PCA** (from 317 → 39 components).  
   - Class imbalance handling (undersampling/oversampling strategies tested).  

2. **Modeling**  
   - Base models: **Logistic Regression**, **Random Forest**, **XGBoost**.  
   - Hyperparameter tuning for each base model.  
   - **Stacking Ensemble**: Logistic Regression as meta-classifier combining predictions of base models.  

3. **Model Interpretation**  
   - **Permutation Importance** to assess feature contributions.  
   - **SHAP analysis** for explainability at both global and local prediction levels.  

---

## 🚀 Results
- **Stacking ensemble achieved ~89% accuracy**, outperforming individual models and boosting generalization.  
- PCA reduced computational complexity without major performance loss.  
- SHAP values highlighted features most associated with customer dissatisfaction, aiding potential business interventions.  

---

## 🛠️ Tech Stack
- **Languages/Libraries:** Python, pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn, SHAP  
- **Techniques:** PCA, Logistic Regression, Random Forest, XGBoost, Stacking Ensemble, SHAP interpretability  

---

## 📈 Key Takeaways
- Ensemble stacking improved prediction robustness compared to boosting/bagging alone.  
- PCA balanced performance with interpretability, reducing dimensions by 87%.  
- SHAP analysis provided actionable insights into top dissatisfaction drivers despite anonymized features.  

---

## 🔮 Future Work
- Test additional ensemble methods (LightGBM, CatBoost).  
- Apply advanced imbalance handling (SMOTE, cost-sensitive learning).  
- Deploy as a web app (Streamlit/Dash) for real-time satisfaction scoring.  

---

## 🙌 Acknowledgments
- Dataset provided by **Santander Bank** via Kaggle.  
- Project developed as part of academic coursework (Northeastern University).  

