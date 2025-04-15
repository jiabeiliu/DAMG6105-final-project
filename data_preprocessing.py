import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

def preprocess_and_pca(filepath):
    df = pd.read_csv(filepath)

    # Step 1: Remove outliers using IQR
    def remove_outliers_iqr(df):
        df_cleaned = df.copy()
        for col in df.columns:
            if df[col].dtype != 'O':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        return df_cleaned

    df = remove_outliers_iqr(df)

    # Step 2: Impute missing values with median
    df = df.fillna(df.median(numeric_only=True))

    # Step 3: Normalize all columns
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Step 4: Unsupervised Learning for Generating Labels
    clustering_features = df_normalized[['Glucose', 'BMI', 'Age']]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(clustering_features)
    diabetes_cluster = np.argmax(kmeans.cluster_centers_[:, 0])
    df_normalized['Outcome'] = (clusters == diabetes_cluster).astype(int)

    # Step 5: Feature Extraction with PCA
    X = df_normalized.drop(columns=['Outcome'])
    y = df_normalized['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pca = PCA(n_components=3)
    X_train_pca = pd.DataFrame(pca.fit_transform(X_train), columns=['PC1', 'PC2', 'PC3'])
    X_test_pca = pd.DataFrame(pca.transform(X_test), columns=['PC1', 'PC2', 'PC3'])

    return X_train_pca, X_test_pca, y_train, y_test
