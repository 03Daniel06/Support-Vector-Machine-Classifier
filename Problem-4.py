import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest






def remove_outliers(X, y, contamination=0.1):
    """Remove outliers from the dataset using Isolation Forest."""
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(X)
    mask = outliers != -1
    return X[mask], y[mask]

# Read data
def read_data(file_path):
    """Reads data from a file and stores it in a matrix."""
    return np.loadtxt(file_path)

def read_labels(file_path):
    """Reads labels from a file and stores them in a vector."""
    return np.loadtxt(file_path).astype(np.int64)

# Training and testing the classifier
def main():
    # Load your data (replace this with your actual data loading)
    X_train = read_data('Data-1-train.txt')  # Feature matrix for training
    y_train = read_labels('Label-1-train.txt')  # Labels for training
    X_test = read_data('Data-1-test.txt')  # Feature matrix for testing
    y_test = read_labels('Label-1-test.txt')  # Labels for testing

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Remove outliers from the test data
    contamination = 0.1  # Tunable parameter
    X_test, y_test = remove_outliers(X_test, y_test, contamination=contamination)

    # Initialize the SVM classifier with RBF kernel and best parameters
    best_svm_classifier = svm.SVC(kernel='rbf', C=4.0, gamma=0.0011500000000003608)

    # Train the SVM classifier with the best parameters
    best_svm_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = best_svm_classifier.predict(X_test)

    # Evaluate the classifier
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
