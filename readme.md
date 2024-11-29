# Naive Bayes Classifier for Iris Dataset

This project implements a **Gaussian Naive Bayes** classifier from scratch to classify iris flower species based on various features such as sepal length, sepal width, petal length, and petal width. The classifier is trained using a training dataset and tested using a separate test dataset. The program calculates prior probabilities, likelihoods for each class, and makes predictions based on Bayes' theorem.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [How the Code Works](#how-the-code-works)

## Project Description

This project builds a **Gaussian Naive Bayes** classifier. The classifier assumes that features are conditionally independent given the class label, and that the distribution of each feature follows a **Gaussian (Normal)** distribution.

- **Class priors**: These are calculated from the training data and represent the probability of each class in the dataset.
- **Likelihoods**: For each class, the program calculates the mean and standard deviation for each feature (sepal length, sepal width, petal length, petal width) to model the Gaussian distribution.
- **Prediction**: Given a new test sample, the classifier calculates the probability of that sample belonging to each class using Bayes' theorem and assigns the class with the highest probability.

## Installation

To run this program, you need to have Python 3.x and the following libraries installed:

- `pandas`: For data manipulation and CSV file reading.
- `numpy`: For numerical operations and math functions.
- `scikit-learn`: For splitting the dataset (optional, depending on how you handle data).

### Install Dependencies

You can install the required dependencies using `pip`:

```bash
pip install pandas numpy scikit-learn
```

## Usage

### Step 1: Prepare the Dataset

You should have two CSV files:

- **`iris_train.csv`**: The training dataset that contains the feature values and corresponding class labels.
- **`iris_test.csv`**: The test dataset used to evaluate the model.

Each CSV file should have the following format:

| sepal_length | sepal_width | petal_length | petal_width | label  |
| ------------ | ----------- | ------------ | ----------- | ------ |
| 5.1          | 3.5         | 1.4          | 0.2         | Setosa |
| 4.9          | 3.0         | 1.4          | 0.2         | Setosa |
| ...          | ...         | ...          | ...         | ...    |

### Step 2: Run the Program

After installing the required dependencies and preparing the dataset, you can run the program using the following command:

```bash
python main.py
```

The program will:

1. Load the training data from `iris_train.csv` and the test data from `iris_test.csv`.
2. Train the Naive Bayes classifier on the training data.
3. Predict the labels for the test data.
4. Calculate and display the accuracy of the model.

### Step 3: View the Output

The program will print out the following:

- Class priors calculated from the training data.
- Likelihoods (mean and standard deviation) for each feature given each class.
- Predictions for each row in the test dataset.
- The overall accuracy of the model on the test data.

Example output:

```
Class Priors: {'Setosa': 0.33, 'Versicolor': 0.33, 'Virginica': 0.33}

Likelihoods for class 'Setosa': {'sepal_length': {'mean': 5.1, 'std': 0.35}, ...}
Likelihoods for class 'Versicolor': {'sepal_length': {'mean': 5.9, 'std': 0.44}, ...}

Predictions: ['Setosa', 'Setosa', 'Versicolor', ...]

Accuracy: 95.0%
```

## File Structure

```
.
├── main.py    # Main program file
├── iris_train.csv              # Training dataset
├── iris_test.csv               # Test dataset
└── README.md                   # This README file
```

### `main.py`

This is the main program file where the Naive Bayes classifier is implemented. It contains functions for:

- **Loading the data**
- **Training the classifier** (calculating priors and likelihoods)
- **Making predictions**
- **Calculating accuracy**

### `iris_train.csv` and `iris_test.csv`

These CSV files contain the training and testing datasets, respectively. The format of the data should be as described above, with features (sepal length, sepal width, petal length, petal width) and a class label (e.g., Setosa, Versicolor, Virginica).

## How the Code Works

1. **Loading the Data**: The `load_data()` function reads the CSV files using `pandas.read_csv()` and loads the data into a DataFrame.
2. **Training the Classifier**: The `train_naive_bayes()` function calculates:
   - **Class priors**: The probability of each class in the training dataset.
   - **Likelihoods**: The mean and standard deviation of each feature for each class.
3. **Making Predictions**: The `predict_naive_bayes()` function applies Bayes' theorem to compute the probability of each class given the features in the test data. It then selects the class with the highest probability as the predicted class.
4. **Calculating Accuracy**: The `calculate_accuracy()` function compares the predicted labels with the true labels and computes the accuracy of the model.

---

### Customizations:

- If you want to use `train_test_split()` to split the dataset into training and testing sets randomly, you can modify the code to load the dataset once and split it using `train_test_split()` instead of providing separate training and test CSV files.

### Notes:

- This implementation assumes that all features are continuous and normally distributed. If you have categorical features, consider using a different approach or pre-processing the data (e.g., using one-hot encoding).
- The model could be improved by applying **Laplace smoothing** for zero probabilities and handling edge cases (e.g., when a feature has no variance for a class).

---
