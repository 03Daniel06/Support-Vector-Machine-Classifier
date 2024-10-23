"""
This is a personal engineering project created by Daniel Northcott, an engineering student with minors in Computer Science and Mathematics. 
Set to graduate in December 2025, Daniel has built this project as part of his learning journey to apply machine learning techniques, 
such as Support Vector Machines (SVMs), to real-world classification tasks. This project includes data preprocessing, 
outlier removal using Isolation Forest, and SVM training and testing on multiple datasets.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest

def remove_outliers(X, y, contamination=0.1):
    """
    Remove outliers from the dataset using Isolation Forest.
    
    Parameters:
    X: Feature matrix
    y: Labels
    contamination: The proportion of outliers in the dataset (default is 0.1)
    
    Returns:
    A tuple (X_cleaned, y_cleaned) containing the dataset without outliers.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    outliers = iso_forest.fit_predict(X)  # Predict outliers (-1 means outlier)
    mask = outliers != -1  # Create mask to filter out the outliers
    return X[mask], y[mask]  # Return cleaned data

# Function to read data from a text file
def read_data(file_path):
    """
    Reads data from a file and stores it in a matrix.
    
    Parameters:
    file_path: Path to the data file
    
    Returns:
    A NumPy array containing the feature data.
    """
    return np.loadtxt(file_path)

# Function to read labels from a text file
def read_labels(file_path):
    """
    Reads labels from a file and stores them in a vector.
    
    Parameters:
    file_path: Path to the label file
    
    Returns:
    A NumPy array containing the label data as integers.
    """
    return np.loadtxt(file_path).astype(np.int64)

# Function to prompt user to select a dataset
def get_user_choice():
    """
    Prompts the user to select which dataset to use for training and testing.
    
    Returns:
    The dataset number selected by the user as a string.
    """
    while True:
        choice = input("Please enter the dataset number you want to use (1-4): ")
        if choice in ['1', '2', '3', '4']:  # Validate input
            return choice
        else:
            print("Invalid input. Please enter a number between 1 and 4.")

# Main function to handle data loading, preprocessing, training, and evaluation
def main():
    """
    Main function to load data, preprocess it, train an SVM classifier, and evaluate it.
    """
    # Get the user's choice of dataset
    dataset_number = get_user_choice()

    # Construct file names for training and testing based on the user's choice
    data_train_file = f'Data-{dataset_number}-train.txt'
    label_train_file = f'Label-{dataset_number}-train.txt'
    data_test_file = f'Data-{dataset_number}-test.txt'
    label_test_file = f'Label-{dataset_number}-test.txt'

    # Load the training and testing datasets
    X_train = read_data(data_train_file)  # Load training feature matrix
    y_train = read_labels(label_train_file)  # Load training labels
    X_test = read_data(data_test_file)  # Load testing feature matrix
    y_test = read_labels(label_test_file)  # Load testing labels

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform the training data
    X_test = scaler.transform(X_test)  # Transform the test data using the same scaling

    # Remove outliers from the test data (this can be tuned with the contamination parameter)
    contamination = 0.1  # 10% of data is expected to be outliers
    X_test, y_test = remove_outliers(X_test, y_test, contamination=contamination)

    # Initialize the SVM classifier with an RBF kernel and previously determined best hyperparameters
    best_svm_classifier = svm.SVC(kernel='rbf', C=4.0, gamma=0.0011500000000003608)

    # Train the classifier on the training data
    best_svm_classifier.fit(X_train, y_train)

    # Make predictions on the cleaned test data
    y_pred = best_svm_classifier.predict(X_test)

    # Output the confusion matrix for evaluation
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Calculate and print the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Entry point of the program
if __name__ == "__main__":
    main()  # Call the main function to run the project
