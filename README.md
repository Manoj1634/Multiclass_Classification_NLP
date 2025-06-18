# Multiclass Text Classification with XGBoost and Naive Bayes

This project focuses on multiclass classification of textual emotion data. The dataset includes labels like:

- anger, boredom, empty, enthusiasm, fun, happiness, hate, love, neutral, relief, sadness, surprise, worry

The dataset is imbalanced and 836555 rows are present in the data, so preprocessing and model choice are critical for better performance.

## ✅ Preprocessing Steps
- Lowercasing
- Removing special characters
- Stopword removal
- Lemmatization

These steps significantly improved classification performance, especially with simple models.

---

## 🔍 Models and Performance

### 1. **Multinomial Naive Bayes** (Baseline + Fast Inference)
- **Before Preprocessing**: ~0.68 metric score (avg of F1-macro & Cohen’s Kappa)
- **After Preprocessing**: **~0.83** metric score ✅

### 2. **XGBoost Classifier**
- Used GPU-accelerated training with class weighting
- Achieved **~0.98** metric score on training data (with early stopping)
![Screenshot](https://github.com/user-attachments/assets/c16fd841-d0f0-4151-9aca-8200f26b465d)

---

## 📦 Output Files

The following files are generated after training:
- `Ivan_model.pickle` – Trained XGBoost model
- `Ivan_vectorizer.joblib` – TF-IDF vectorizer
- `Ivan_labelencoder.joblib` – Label encoder

---

## 💡 Quick Note

While XGBoost achieves very high accuracy, Multinomial Naive Bayes is extremely fast.

---

