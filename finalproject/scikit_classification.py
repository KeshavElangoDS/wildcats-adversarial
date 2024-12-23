"""
scikit_classification.py: Performs randomForest classification on input data

This module contains functions to perform randomForest classification on input data.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def extract_image_features(image_path):
    """
    Extract mean and variance for RGB channels from an image.

    Args:
    - image_path (str): Path to the image file.

    Returns:
    - features (list): List of features [mean_R, mean_G, mean_B, var_R, var_G, var_B].
    """
    img = Image.open(image_path)
    img = img.convert("RGB")

    img_array = np.array(img)
    mean = img_array.mean(axis=(0, 1))
    variance = img_array.var(axis=(0, 1))

    return list(mean) + list(variance)


def extract_features_from_dataframe(dataset_folder_absolute_path, df):
    """
    Extract features from a dataframe of image file paths.

    Args:
        dataset_folder_absolute_path (path): Absolute path of the folder containing data.
        df (pd.DataFrame): DataFrame containing image paths and labels.

    Returns:
        features (list): List of extracted features for each image.
        labels (list): List of corresponding labels.
    """
    features = []
    labels = []

    for _, row in df.iterrows():

        image_path = os.path.join(dataset_folder_absolute_path, row["filepaths"])
        feature_vector = extract_image_features(image_path)

        features.append(feature_vector)
        labels.append(row["class id"])

    return np.array(features), np.array(labels)


def train_and_evaluate_randomforest(
    X_train, y_train, X_test, y_test, n_estimators=100, random_state=42
):
    """
    Train a Random Forest Classifier, make predictions, and
    evaluate accuracy on both training and test sets.

    Args:
    - X_train (array-like): Feature matrix for the training set.
    - y_train (array-like): Labels for the training set.
    - X_test (array-like): Feature matrix for the test set.
    - y_test (array-like): Labels for the test set.
    - n_estimators (int): The number of trees in the forest (default is 100).
    - random_state (int): Random seed for reproducibility (default is 42).

    Returns:
    - None: prints the training and testing accuracy.
    """

    classifier = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    y_train_pred = classifier.predict(X_train)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def train_and_tune_randomforest_GridSearch(
    X_train,
    y_train,
    X_test,
    y_test,
    param_grid,
    cv=5,
    n_jobs=-1,
    verbose=0,
    random_state=42,
):
    """
    Train a Random Forest Classifier, perform Grid Search to find the best hyperparameters,
    and evaluate on the test set.

    Args:
    - X_train (array-like): Feature matrix for the training set.
    - y_train (array-like): Labels for the training set.
    - X_test (array-like): Feature matrix for the test set.
    - y_test (array-like): Labels for the test set.
    - param_grid (dict): Dictionary with hyperparameters to tune in GridSearchCV.
    - cv (int): Number of cross-validation folds (default is 5).
    - n_jobs (int): Number of jobs to run in parallel (default is -1, which uses all CPUs).
    - verbose (int): Verbosity level (default is 0, no output).
    - random_state (int): Random seed for reproducibility (default is 42).

    Returns:
    - None: prints the Best parameters and test accuracy
    """
    classifier = RandomForestClassifier(random_state=random_state)

    grid_search = GridSearchCV(
        classifier, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    # Get the best RandomForest model from the grid search
    best_randomforest_classifier = grid_search.best_estimator_
    best_randomforest_classifier.fit(X_train, y_train)

    y_pred = best_randomforest_classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
