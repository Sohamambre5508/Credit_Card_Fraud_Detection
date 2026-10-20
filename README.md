# 💳 Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using **Logistic Regression**, built to handle extreme class imbalance (only 0.172% fraud).

## 📌 Problem Statement
Credit card fraud costs billions every year. The challenge is that fraud is extremely rare, so a model that labels every transaction "legit" would be 99.8% accurate and still catch zero fraud. This project focuses on detecting fraud properly, not just scoring high accuracy.

## 📊 Dataset
- **Source:** [Kaggle – Credit Card Fraud Detection (ULB)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Transactions by European cardholders in **September 2013**, spanning two days
- **284,807 transactions**, of which **492 are fraud (0.172%)**
- Features `V1`–`V28` are PCA-transformed (anonymized for confidentiality)
- `Time` = seconds since the first transaction, `Amount` = transaction amount
- `Class` = target (1 = fraud, 0 = legit)

> The dataset is not included in this repo due to size. Download it from the link above and place `creditcard.csv` in the `data/` folder.

## ⚙️ Approach
1. Data exploration and class distribution analysis
2. Preprocessing (scaling `Amount` and `Time`)
3. Handling class imbalance using **[undersampling / SMOTE / class_weight – fill in what you used]**
4. Training a **Logistic Regression** model
5. Evaluation using metrics suited to imbalanced data

## 🏆 Results
| Model | [METRIC] |
|-------|----------|
| Logistic Regression | **96.69%** |


## 🛠️ Tech Stack
Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## 🔮 Future Improvements
- Try Random Forest and XGBoost and compare results
- Evaluate with the Precision-Recall curve (AUPRC)
- Threshold tuning to balance false alarms against missed fraud
- Deploy as a Streamlit web app

## 👤 Author
**[Soham Ambre]** · [LinkedIn](your-link) · [GitHub](your-link)
