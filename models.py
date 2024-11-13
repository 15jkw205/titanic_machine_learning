# Jakob West & Justin Landry
# 11/12/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# models.py


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import catboost as cb

# Convert data to NumPy arrays and handle NaN values
def convert_to_numpy(x, y):
    if isinstance(x, pd.DataFrame):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)
    return x, y

# Split data into training and validation sets (80/20 split)
def split_data(x, y, test_size=0.2, random_state=0):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

# Core Models
def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=0, solver='liblinear')
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier(max_depth=5, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_xgboost(x_train, y_train):
    # Convert DataFrame to NumPy array if needed
    if isinstance(x_train, pd.DataFrame):
        x_train = x_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Ensure there are no NaN or infinite values in the data
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Initialize the XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        eval_metric='logloss'
    )

    try:
        model.fit(x_train, y_train)
    except ValueError as e:
        print(f"Error training XGBoost: {e}")
        print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
        raise e

    return model

def train_catboost(x_train, y_train):
    model = cb.CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_state=0, verbose=0)
    model.fit(x_train, y_train)
    return model

def train_hist_gradient_boosting(x_train, y_train):
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(x_train, y_train)
    return model

def train_voting_classifier(x_train, y_train):
    model = VotingClassifier(
        estimators=[
            ('rf', train_random_forest(x_train, y_train)),
            ('xgb', train_xgboost(x_train, y_train)),
            ('cat', train_catboost(x_train, y_train))
        ],
        voting='soft'
    )
    model.fit(x_train, y_train)
    return model

def train_stacking_classifier(x_train, y_train):
    model = StackingClassifier(
        estimators=[
            ('rf', train_random_forest(x_train, y_train)),
            ('xgb', train_xgboost(x_train, y_train)),
            ('cat', train_catboost(x_train, y_train))
        ],
        final_estimator=LogisticRegression()
    )
    model.fit(x_train, y_train)
    return model

# Function to train and evaluate all models using 80/20 split
def run_all_models(x_train, x_val, y_train, y_val):
    """Train all models, evaluate performance on validation set, and select the best model."""
    accuracy_results = {}
    best_accuracy = 0
    best_model = None
    best_model_name = None

    models = {
        'Logistic Regression': train_logistic_regression,
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest,
        'XGBoost': train_xgboost,
        'CatBoost': train_catboost,
        'Hist Gradient Boosting': train_hist_gradient_boosting,
        'Voting Classifier': train_voting_classifier,
        'Stacking Classifier': train_stacking_classifier
    }

    for model_name, train_func in models.items():
        print(f"\nTraining {model_name}...")
        try:
            model = train_func(x_train, y_train)
            accuracy = evaluate_model(model, x_val, y_val)
            accuracy_results[model_name] = accuracy

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

            print(f"{model_name} Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")

    return accuracy_results, best_model_name, best_model

# Function to evaluate model on validation set
def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    return accuracy_score(y_val, y_pred)