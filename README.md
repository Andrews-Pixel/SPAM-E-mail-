# 📧 Spam Email Classifier — Naive Bayes (UCI Spambase Dataset)

A machine learning project that classifies emails as **Spam** or **Ham (Not Spam)** using Gaussian Naive Bayes on the UCI Spambase dataset.


## 📌 Project Overview

Email spam detection is one of the classic problems in machine learning. This project builds a spam classifier from scratch — starting with raw data loading, exploratory data analysis, preprocessing, model training, evaluation, and iterative improvement through hyperparameter tuning and feature transformation.


## 📂 Dataset

**Source:** [UCI Machine Learning Repository — Spambase](https://archive.ics.uci.edu/dataset/94/spambase)
**Total Records:** 4,601 emails
**Features:** 57 numeric features (word frequencies, character frequencies, capital run lengths)
**Target:** `is_spam` → 1 (Spam), 0 (Ham)

| Class | Count |
|---|---|
| Ham (0) | 2,788 |
| Spam (1) | 1,813 |


## 🛠️ Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn


## 🔄 Project Workflow

Raw Data Loading
      ↓
Exploratory Data Analysis (EDA)
      ↓
Preprocessing (StandardScaler)
      ↓
Model Training (GaussianNB)
      ↓
Evaluation (Confusion Matrix, Classification Report)
      ↓
Hyperparameter Tuning (GridSearchCV)
      ↓
Feature Transformation (Log Transform)
      ↓
Final Evaluation



## 📊 Model Evolution & Results

| Stage       | Approach                            | Test Accuracy |
| Baseline    | GaussianNB (default)                | 82.19% |
| Tuning      | GridSearchCV (`var_smoothing=1e-7`) | 82.19% |
| Improvement | Log Transform + GaussianNB          | **83.93%** |


## 🔍 Key Steps Explained

### 1. Data Loading
Column names were parsed programmatically from `spambase.names` — avoiding hardcoding 57 column names manually.

### 2. EDA
- Checked for null values → None found
- Checked class distribution → mild imbalance (60% Ham / 40% Spam)
- Plotted full correlation heatmap (58×58)

### 3. Preprocessing
Applied `StandardScaler` to normalize all 57 features before training.

### 4. Hyperparameter Tuning
Used `GridSearchCV` with 5-fold cross validation to find the best `var_smoothing`:
python
param = {'var_smoothing': [1e-11, 1e-9, 1e-7, 1e-5, 1e-3]}

Best value found: `1e-7`

### 5. Log Transform
Applied `np.log1p()` to fix right-skewed word frequency distributions before scaling — bringing the data closer to the Gaussian distribution that GaussianNB expects.


## 📈 Final Classification Report (Log Transform Model)

              precision    recall  f1-score   support

           0       0.94      0.77      0.85       531
           1       0.75      0.94      0.83       390

    accuracy                           0.84       921
   macro avg       0.85      0.85      0.84       921
weighted avg       0.86      0.84      0.84       921



## 🔢 Confusion Matrix
```
                  | Predicted Ham | Predicted Spam |
| **Actual Ham**  | 391           | 140            |
| **Actual Spam** | 24            | 366            |
```

## ▶️ How To Run

1. Clone the repository:
```bash
git clone https://github.com/Andrews-Pixel/spam_email_detection(Naive bayse).git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook spam_classifier_naive_bayes.ipynb
```

## 👤 Author

**Andrews Martin**
LinkedIn: [andrewsmartin30](https://linkedin.com/in/andrewsmartin30)
