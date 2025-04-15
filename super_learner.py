import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

def run_super_learner(X_train_pca, y_train, X_test_pca, y_test):
    # Initialize base classifiers
    nb = GaussianNB()
    knn = KNeighborsClassifier(n_neighbors=5)
    nn = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

    # Cross-validated predictions for training set
    nb_preds = cross_val_predict(nb, X_train_pca, y_train, cv=5, method='predict')
    knn_preds = cross_val_predict(knn, X_train_pca, y_train, cv=5, method='predict')
    nn_preds = cross_val_predict(nn, X_train_pca, y_train, cv=5, method='predict')

    # Meta features for meta learner
    meta_features = pd.DataFrame({
        'NB': nb_preds,
        'KNN': knn_preds,
        'NN': nn_preds
    })

    # Train meta learner
    meta_learner = DecisionTreeClassifier(random_state=42)
    meta_learner.fit(meta_features, y_train)

    # Refit base learners on full training set
    nb.fit(X_train_pca, y_train)
    knn.fit(X_train_pca, y_train)
    nn.fit(X_train_pca, y_train)

    # Predict meta features for test set
    test_meta_features = pd.DataFrame({
        'NB': nb.predict(X_test_pca),
        'KNN': knn.predict(X_test_pca),
        'NN': nn.predict(X_test_pca)
    })

    # Final predictions from meta learner
    final_predictions = meta_learner.predict(test_meta_features)
    accuracy = accuracy_score(y_test, final_predictions)

    print(f"Super Learner Accuracy on Test Set: {accuracy:.4f}")
    return accuracy
