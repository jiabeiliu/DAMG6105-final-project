from data_preprocessing import preprocess_and_pca
from super_learner import run_super_learner

def main():
    # Step 1–3: Preprocessing and PCA
    X_train_pca, X_test_pca, y_train, y_test = preprocess_and_pca("diabetes_project.csv")

    # Step 4: Super Learner Classification
    run_super_learner(X_train_pca, y_train, X_test_pca, y_test)

if __name__ == "__main__":
    main()
