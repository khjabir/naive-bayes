import pandas as pd
import numpy as np
from math import log, sqrt, pi

# Function to load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Function to train Naive Bayes classifier
def train_naive_bayes(data):
    X = data.drop(columns=['label'])  # Features
    y = data['label']  # Target label
    
    # Calculate class priors
    class_priors = y.value_counts(normalize=True).to_dict()
    print("\nClass Priors:", class_priors)  # Debugging output

    # Calculate likelihoods for each class
    likelihoods = {}
    for label in class_priors:
        class_data = X[y == label]
        print(f"\n{label} Class Data:\n{class_data}\n")
        likelihoods[label] = {}
        for column in X.columns:
            mean = class_data[column].mean()
            std = class_data[column].std()
            likelihoods[label][column] = {'mean': mean, 'std': std}
        
        print(f"\nLikelihoods for class '{label}':\n{likelihoods[label]}")  # Debugging output

    return class_priors, likelihoods

# Function to predict using Naive Bayes
def predict_naive_bayes(class_priors, likelihoods, test_data):
    predictions = []
    for _, row in test_data.iterrows():
        class_probs = {}
        for label in class_priors:
            prob = log(class_priors[label])  # Log of prior probability
            for column in test_data.columns:
                if column in likelihoods[label]:  # Ensure column exists in likelihoods
                    mean = likelihoods[label][column]['mean']
                    std = likelihoods[label][column]['std']
                    if std > 0:  # Avoid division by zero
                        prob += log(1 / (std * sqrt(2 * pi))) - ((row[column] - mean) ** 2) / (2 * std ** 2)
            
            class_probs[label] = prob
        
        print(f"\nClass probabilities for row {row.name}: {class_probs}")  # Debugging output
        predicted_label = max(class_probs, key=class_probs.get)
        print(f"Max Value: {predicted_label}")
        predictions.append(predicted_label)
    
    return predictions

# Function to calculate accuracy
def calculate_accuracy(predictions, true_labels):
    correct = sum(np.array(predictions) == np.array(true_labels))
    accuracy = correct / len(true_labels) * 100
    return accuracy

# Main function
def main():
    # Load the training dataset
    df_train = pd.read_csv('iris_train.csv')  # Load your training data here
    
    # Load the test dataset
    df_test = pd.read_csv('iris_test.csv')  # Load your test data here
    
    # Train Naive Bayes classifier on the training data
    class_priors, likelihoods = train_naive_bayes(df_train)
    
    # Separate features and labels for the test data
    test_features = df_test.drop(columns=['label'])

    print(f"\nTest Data: {df_test.head()}")

    test_labels = df_test['label']
    
    # Predict labels for the test data
    predictions = predict_naive_bayes(class_priors, likelihoods, test_features)
    print(f"\nPredictions: {predictions}\n")
    
    # Calculate accuracy
    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"\n\nAccuracy: {accuracy:.2f}%")

# Run the program
if __name__ == "__main__":
    main()
