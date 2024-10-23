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

def get_user_choice():
    while True:
        choice = input("Please enter the dataset number you want to use (1-4): ")
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("Invalid input. Please enter a number between 1 and 4.")


# Training and testing the classifier
def main():
    # Get the user's choice
    dataset_number = get_user_choice()

    # Construct file names based on the user's choice
    data_train_file = f'Data-{dataset_number}-train.txt'
    label_train_file = f'Label-{dataset_number}-train.txt'
    data_test_file = f'Data-{dataset_number}-test.txt'
    label_test_file = f'Label-{dataset_number}-test.txt'

    # Read the data and labels
    X_train = read_data(data_train_file)  # Feature matrix for training
    y_train = read_labels(label_train_file)  # Labels for training
    X_test = read_data(data_test_file)  # Feature matrix for testing
    y_test = read_labels(label_test_file)  # Labels for testing


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
