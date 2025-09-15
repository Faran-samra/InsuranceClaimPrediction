# Insurance Claim Prediction

This project predicts whether a person is likely to claim insurance using machine learning models, including **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost**.  

It demonstrates **data preprocessing, feature engineering, model training, evaluation, and single prediction functionality**.  

---

## **Project Structure**



---

## **Features**

- Data preprocessing with **ColumnTransformer** (numeric & categorical pipelines)  
- Feature engineering: `bmi_category` and `age_bucket`  
- Model training with **GridSearchCV** for hyperparameter tuning  
- Evaluation metrics:
  - **Accuracy**  
  - **F1-score**  
  - **Precision & Recall**  
  - **ROC-AUC**  
  - **Confusion matrix & classification report**  
- Save/load trained models with **Joblib**  
- Single prediction function for **Streamlit or Flask apps**

---

## **Getting Started**

### 1. Clone the repository

```bash
git clone git@github.com:Faran-samra/InsuranceClaimPrediction.git
cd InsuranceClaimPrediction

2. Install dependencies

pip install -r requirements.txt

3. Run the Jupyter notebook

jupyter notebook Insurance.ipynb

