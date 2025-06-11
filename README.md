```markdown
# Credit Card Approval Prediction with LightGBM and XGBoost

This project is a machine learning pipeline for predicting credit card approvals using structured data from application and credit records.
It leverages two state-of-the-art gradient boosting frameworks—**LightGBM** and **XGBoost**—with hyperparameter optimization via **Optuna**,
detailed preprocessing utilities, and rich performance tracking.

---

## 📁 Project Structure

---

## 🧠 Objective

The goal is to build an accurate binary classifier that predicts whether a credit card application should be **approved or rejected** based on applicant data and historical credit behavior.

---

## 🧾 Datasets

- **application_record.csv**: Contains demographic and application-related information about applicants.
- **credit_record.csv**: Time-series data indicating historical credit behavior and status over time.

> Both datasets are used in tandem during preprocessing to engineer the final feature set.

---

## 🔧 Features

- Feature engineering with categorical encoding and merging of time series data
- Model tuning via **Optuna** for both LightGBM and XGBoost
- Evaluation through cross-validation and saved logs
- Model saving and reusability (`.pkl` files)
- Visualization scripts to assist in understanding model decisions and distributions

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
````

Or manually install:

```bash
pip install lightgbm xgboost optuna pandas numpy scikit-learn matplotlib seaborn
```

---

### 2. Preprocess Data

```bash
# Data preparation utilities
python helper_functions/preprocessing.py
```

---

### 3. Train Models

#### LightGBM

```bash
python lightgbm_model/final_train.py
```

#### XGBoost

```bash
python xgboost_model/train.py
```

---

### 4. Evaluate & Log Results

After training, evaluation metrics are automatically saved in:

* `metrics/XGBoost_performance.log`
* `metrics/lightGBM_performance.log`

---

### 5. Test Models

```bash
# LightGBM
python lightgbm_model/testing.py

# XGBoost
python xgboost_model/test.py
```

---

## 📊 Analysis

Visual exploration scripts are located in the `analysis/` folder. This includes:

* Distribution plots
* Boxplots for feature distributions
* Performance plots and custom EDA

---

## 📦 Model Files

* `xgboost_credit_card_approval.pkl`: Saved XGBoost model
* `lightgbm_credit_card_approval.pkl`: Saved LightGBM model

These can be directly loaded for inference or deployment.

---

## 📈 Optimization

Hyperparameter tuning is done using **Optuna**. You can trigger optimization runs via:

```bash
python lightgbm_model/optuna_optimization.py
python xgboost_model/optuna_optimization.py
```

---

## ✍️ Author

Cliff – Computer Vision and Machine Learning Engineer

---

## 📌 To-Do (Optional)

* [ ] Add Streamlit/Gradio UI for live prediction
* [ ] Implement model version tracking (e.g., MLflow)
* [ ] Deploy via FastAPI or Flask
* [ ] Unit tests for helper functions

---

## 📄 License

This project is licensed under the MIT License.

```
