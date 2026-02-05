# 🚨 Fraud Detection in Financial Transactions

This repository contains a **machine learning project** focused on detecting fraudulent transactions using real transaction data. The goal is to build a predictive model that accurately identifies fraud while demonstrating a complete end-to-end data science workflow — from exploratory analysis to model training and evaluation.

---

## 🔍 Project Overview

Fraudulent transactions cause significant financial loss for businesses and customers. In this project, we analyze transaction records to identify patterns and build models to predict whether a transaction is fraudulent. The project includes data cleaning, feature engineering, visualization, and building a machine learning pipeline.

---

## 📂 Dataset Description

The dataset contains transaction records with the following types of features:

- Transaction amount
- Account balances (before and after transactions)
- Transaction type
- Fraud label (`isFraud`)
- Other transaction attributes

Each instance represents a single transaction with a binary target variable indicating whether it is fraudulent.

---

## 📊 Exploratory Data Analysis (EDA)

We perform several EDA steps to understand the data distribution and relationships:

- Visualizing class distribution (`isFraud`)
- Understanding transaction types
- Analyzing balance differences
- Transaction amount distribution (log scale)
- Box plots to compare fraudulent vs. non-fraudulent transactions
- Heatmap for correlation analysis
- Identifying zero balance anomalies after transfers

Visualizations help reveal patterns and potential fraud indicators.

---

## 🛠 Data Preprocessing & Feature Engineering

- Handled null values and duplicates
- Created new features like balance differences
- Dropped irrelevant features
- One-Hot Encoding for categorical variables
- Standard scaling for numeric features

---

## 🤖 Machine Learning Model

We build an ML pipeline using:
- Preprocessing with `ColumnTransformer`
- Scaler for numeric features
- OneHotEncoder for categorical features
- A balanced **Logistic Regression** model

We also trained a **KNN model** for comparison.

### Models Used
- Logistic Regression (primary model)
- K-Nearest Neighbors (comparison)

---

## 🧠 Model Evaluation

The pipeline is trained and evaluated using:
- Train-test split
- Classification report (Precision, Recall, F1-Score)
- Confusion Matrix
- Accuracy score for performance

The best model was saved using **joblib** for deployment.

Conclusion

This project demonstrates a full ML workflow for fraud detection using real data, robust preprocessing, visualization, and model building. It’s a strong portfolio piece for anyone applying to data science or machine learning roles.
