# 🏦 Credit Scoring Models — Loan Approval Prediction with Machine Learning

> Predicting whether a loan application should be approved by comparing four machine learning models, with the best one served through a Flask web app.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?logo=flask&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-UI-38B2AC?logo=tailwindcss&logoColor=white)

---

## 📌 Overview

Banks have to decide quickly and fairly who should get a loan. Traditionally this relies on loan officers applying the *5C principles* (Character, Capital, Capacity, Collateral, Conditions), which is slow and open to bias.

This project builds a **data-driven credit scoring system** that:

1. Cleans and transforms real loan application data
2. Trains and compares **four classifiers**: Random Forest, Decision Tree, Logistic Regression and an MLP Neural Network
3. Evaluates them on **accuracy, precision, recall, F1-score, confusion matrix and ROC-AUC**
4. Deploys the best model (**Random Forest**) behind a **Flask web app** where a user fills in an applicant's details and gets an instant approval prediction

This was developed as a final-year B.E. project in Electronics & Telecommunication Engineering (Goa University, 2022–23). The full write-up is in [`Report.pdf`](Report.pdf).

---

## 📊 Results

All models were trained on the same 75 / 25 train-test split (`random_state=42`, 154 test applications).

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| 🌲 **Random Forest** | **79.2%** | **76.6%** | **98.0%** | **86.0%** | **0.79** |
| Logistic Regression | 77.3% | 74.8% | 98.0% | 84.8% | 0.73 |
| MLP Neural Network (100, 50) | 76.6% | 74.6% | 97.0% | 84.3% | 0.76 |
| Decision Tree | 73.4% | 77.1% | 84.0% | 80.4% | 0.69 |

**🏆 Random Forest** scored highest on accuracy, F1 and ROC-AUC, so it was saved as `model.pkl` and used in the web app.

**Insight:** every model has very high recall on approved loans but more trouble with rejections (Random Forest: 24 true negatives vs 30 false positives). The dataset is imbalanced (422 approved vs 192 rejected), which makes this the main thing to improve next (see [Future Improvements](#-future-improvements)).

---

## 🗂️ Dataset

The project uses the public **Loan Prediction** dataset:

| File | Rows | Description |
|---|---:|---|
| `train.csv` | 614 | Labelled applications (`Loan_Status` = Y / N) |
| `test.csv` | 367 | Unlabelled applications for inference |

**Features**

| Type | Columns |
|---|---|
| Categorical | `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`, `Credit_History` |
| Numerical | `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term` |
| Target | `Loan_Status` |

---

## ⚙️ ML Pipeline

```
Raw CSV ─► Missing-value imputation ─► EDA ─► Feature engineering ─► One-hot encoding ─► Train/test split ─► 4 models ─► Evaluation ─► Best model → model.pkl ─► Flask app
```

**1. Data cleaning**
- Numerical gaps (`LoanAmount`, `Loan_Amount_Term`, `Credit_History`) filled with the **mean**
- Categorical gaps (`Gender`, `Married`, `Dependents`, `Self_Employed`) filled with the **mode**

**2. Exploratory data analysis**
- Count plots for categorical features and distribution plots for income and loan amounts, which show a strong right skew

**3. Feature engineering**
- New feature: **`Total_Income` = ApplicantIncome + CoapplicantIncome**
- **Log transforms** on `ApplicantIncome`, `LoanAmount`, `Loan_Amount_Term` and `Total_Income` to reduce skew and outlier impact
- Raw columns and `Loan_ID` dropped after transformation

**4. Encoding**
- One-hot encoding (`drop_first=True`), giving **14 model inputs**

**5. Modelling and evaluation**
- Random Forest, Decision Tree, Logistic Regression (`max_iter=1000`) and MLP (`hidden_layer_sizes=(100, 50)`)
- Each model gets a confusion matrix, accuracy, precision, recall, F1 and a plotted ROC curve with AUC

---

## 🌐 Web Application

A lightweight **Flask** app serves the trained Random Forest model.

- **Home page (`index.html`):** landing page with a call to action
- **Prediction page (`prediction.html`):** form for gender, marital status, dependents, education, self-employment, credit history, property area, incomes, loan amount and term
- **`/predict` endpoint:** applies the same encoding and log transforms used in training, runs the model and returns **"loan status is Yes / No"**

The UI is built with **Tailwind CSS** and **Bootstrap 5**.

---

## 📁 Project Structure

```
Credit_Scoring_Models/
├── loan model.ipynb    # Data cleaning, EDA, feature engineering, training & evaluation
├── app.py              # Flask app serving the Random Forest model
├── model.pkl           # Trained Random Forest model
├── index.html          # Landing page
├── prediction.html     # Loan application form + result
├── train.csv           # Labelled training data
├── test.csv            # Unlabelled test data
├── Report.pdf          # Full project report (Goa University, 2022–23)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/shubham9448/Credit_Scoring_Models.git
cd Credit_Scoring_Models
```

### 2. Install dependencies
```bash
pip install flask numpy pandas scikit-learn matplotlib seaborn
```

### 3. Put the HTML pages where Flask looks for them
Flask's `render_template` looks for pages inside a `templates/` folder:
```bash
mkdir templates
mv index.html prediction.html templates/
```

### 4. Run the app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser, click **Prediction**, fill in the form and submit.

> **Note:** On Flask 2.3 and newer, `escape` can no longer be imported from `flask`. If you get an `ImportError`, change the first import in `app.py` to `from flask import Flask, request, render_template`.

### 5. Explore or retrain the models
```bash
jupyter notebook "loan model.ipynb"
```
Re-running the notebook regenerates `model.pkl`, which also keeps the model compatible with your installed scikit-learn version.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (RandomForest, DecisionTree, LogisticRegression, MLPClassifier) |
| Model persistence | Pickle |
| Backend | Flask |
| Frontend | HTML, Tailwind CSS, Bootstrap 5 |

---

## 🔮 Future Improvements

- **Handle class imbalance** with class weights or SMOTE to catch more risky applicants (fewer false approvals)
- **Hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV` and k-fold cross-validation
- Try **gradient boosting** models (XGBoost, LightGBM, CatBoost)
- **Explainability** with SHAP or feature importance, so each decision can be justified to the applicant
- Wrap preprocessing and the model in a single **scikit-learn `Pipeline`** so training and serving logic can't drift apart
- Add input validation, a `requirements.txt`, and deploy to Render / Railway / Hugging Face Spaces

---

## 👥 Team

Final-year project, **Padre Conceicao College of Engineering, Goa** (Goa University, 2022–23)

- **Shubham Naik**
- Sagar Singh
- Arya Karapurkar
- Fouziya Nagarchi

**Guide:** Mrs. Rohini H. Korti, Assistant Professor, Department of Electronics & Telecommunication Engineering

---

## 📬 Contact

**Shubham Naik**
[GitHub](https://github.com/shubham9448) · [LinkedIn](https://linkedin.com/in/shubham-naik)

⭐ If you found this project useful, consider giving it a star!
