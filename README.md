👩‍💻 Author
Zainab Cheema & Jiabei Liu
# 🩺 Diabetes Clustering & PCA Analysis

This project demonstrates a semi-supervised learning approach for classifying diabetes outcomes using unsupervised clustering (KMeans) and dimensionality reduction (PCA). The objective is to simulate a classification task by generating synthetic labels from patterns in the data.

---

## 📁 Dataset

The dataset used is `diabetes_project.csv`, which contains medical attributes such as:

- Glucose
- BMI
- Age
- Insulin
- Skin Thickness
- Blood Pressure
- Pregnancies
- Diabetes Pedigree Function

---

## ⚙️ How It Works

1. **Data Preprocessing**  
   - Outliers are removed.
   - Missing values are imputed.
   - Features are normalized.

2. **Label Generation with KMeans**  
   - Unsupervised clustering is applied on selected features (Glucose, BMI, Age).
   - Clusters are interpreted as:
     - `1`: Diabetes (higher average Glucose)
     - `0`: No Diabetes

3. **Dimensionality Reduction using PCA**  
   - Principal Component Analysis is applied to reduce feature space to 3 components.

4. **Train/Test Split**  
   - The dataset is split (e.g., 80% training, 20% testing) after transformation.

5. **Classification**  
   - A super learner model uses base classifiers (Naïve Bayes, KNN, Neural Network) and a Decision Tree meta-learner to classify based on the generated labels.

---

## 🚀 Usage

```python
from your_script_name import preprocess_and_pca

X_train_pca, X_test_pca, y_train, y_test = preprocess_and_pca("data/diabetes.csv")

finalproject/
├── data_preprocessing.py         # Preprocessing + PCA + KMeans
├── super_learner.py              # Ensemble model training
├── stroke_dataset_adaptation.py # (optional transfer learning)
├── main.py                       # Run end-to-end pipeline
├── diabetes_project.csv          # Dataset
├── README.md
└── requirements.txt
