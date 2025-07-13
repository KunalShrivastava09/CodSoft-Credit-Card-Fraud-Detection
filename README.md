# CodSoft-Credit-Card-Fraud-Detection
# CodSoft - Credit Card Fraud Detection 💳🕵️‍♂️

This project was completed as part of the **CodSoft Internship Program**. It uses a supervised machine learning approach (Random Forest Classifier) to detect fraudulent transactions in a highly imbalanced credit card dataset.

---

## 📊 Overview

- **Goal**: Classify transactions as fraudulent (1) or legitimate (0)
- **Challenge**: Extremely imbalanced dataset — frauds represent only ~0.17%
- **Approach**:
  - Data preprocessing & scaling
  - Train-test split
  - RandomForestClassifier
  - Model evaluation using F1 Score, ROC AUC, Confusion Matrix

---

## 📁 Files Included

| File Name                        | Description                                 |
|----------------------------------|---------------------------------------------|
| `CodSoft_CreditCardFraudDetection.ipynb` | Jupyter notebook with full project code |
| `creditcard_sample.csv`          | Sample of original dataset (for testing)   |
| `credit_fraud_model.pkl`         | Trained model file (optional)              |

---

## 📥 Dataset Source

This dataset was originally published by ULB and is available on Kaggle:

📌 [Download Dataset from Kaggle](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqbjhGeGJKaHhLOU5ZY01VbjdkdDVNMWtKZXhBQXxBQ3Jtc0trMERldWpyclA1aTI3R0hTeHdXazRaSndzMEc3T0paZElrNXpCVzhnSEhCVjdUTXRPcXNfdFEyYVhTMFNjN1F3TVB0Y3JFLWVQa1ZwWGpCWGx4UnBEbmJLQWNSa1BXbHN0QndjczRLT0RPdXlhY1dRSQ&q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnelgiriyewithana%2Fcredit-card-fraud-detection-dataset-2023%2Fdata&v=oKgPU8CDON4)

If the full dataset exceeds GitHub's upload limits, only a sample (`creditcard_sample.csv`) is included here for demonstration purposes.

---

## 🧪 Technologies Used

- Python 3.x
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📈 Evaluation Metrics

- ✅ Accuracy
- ✅ Precision, Recall
- ✅ F1 Score
- ✅ ROC AUC Score
- ✅ Confusion Matrix, ROC Curve

---

## 📌 Results (Example)

| Metric      | Score |
|-------------|-------|
| F1 Score    | 0.92  |
| ROC AUC     | 0.98  |

---

## 🔍 Future Improvements

- Use SMOTE for balancing the dataset
- Try alternative models (XGBoost, LightGBM)
- Deploy the model using Streamlit or Flask

---

## 📝 Submission

- **Program**: CodSoft Internship (Machine Learning Track)
- **Task**: Credit Card Fraud Detection
- **Submitted by**: Kunal Shrivastava
- 📅 **Date**: July 2025


