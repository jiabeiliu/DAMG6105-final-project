import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from super_learner import run_super_learner

def preprocess_stroke_dataset(filepath):
    df = pd.read_csv(filepath)

    # Drop rows with missing values in important numeric columns
    df = df.dropna(subset=['bmi'])

    # Encode categorical columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Normalize all columns except target
    features = df.drop(columns=['stroke'])
    target = df['stroke']
    scaler = MinMaxScaler()
    features_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # PCA
    pca = PCA(n_components=3)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train), columns=['PC1', 'PC2', 'PC3'])
    X_test_pca = pd.DataFrame(pca.transform(X_test), columns=['PC1', 'PC2', 'PC3'])

    return X_train_pca, X_test_pca, y_train, y_test


def deploy_on_stroke():
    X_train_pca, X_test_pca, y_train, y_test = preprocess_stroke_dataset("brain_stroke.csv")
    run_super_learner(X_train_pca, y_train, X_test_pca, y_test)

if __name__ == "__main__":
    deploy_on_stroke()
