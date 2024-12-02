# Titanic Machine Learning

Project for CS 3820 - Intro to AI with Justin Landry and Jakob West, team 7. This project is a submission for the Titanic Machine Learning competition on Kaggle. The goal is to build a predictive model to determine if a passenger survived the Titanic disaster based on various features available in the dataset.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Results](#results)

## Overview
The Titanic Machine Learning competition is a beginner-friendly challenge hosted by Kaggle that involves building a model to predict passenger survival on the RMS Titanic. By working on this project, we learned the basics of data analysis, feature engineering, and machine learning model building and evaluation.

In this project, we explored various machine learning models, including Naive Bayes, Logistic Regression, Random Forest, and ensemble methods such as Voting Classifier, to identify the best approach to predict survival outcomes.

## Dataset
The dataset contains the following features:
- **Pclass**: Passenger class (1st, 2nd, or 3rd)
- **Sex**: Gender of the passenger
- **Age**: Age of the passenger
- **SibSp**: Number of siblings or spouses aboard the Titanic
- **Parch**: Number of parents or children aboard the Titanic
- **Fare**: Passenger fare
- **Embarked**: Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)

Target variable:
- **Survived**: Whether the passenger survived (1) or not (0)

The dataset is divided into a training set (`train.csv`) and a test set (`test.csv`). The training set is used for model training and evaluation, and the test set is used for the final model predictions.
 the .csv's can be found in `data/processed`

## Project Structure
The project follows a modular approach with scripts organized as follows:

- **config.py**: Stores project configuration variables, including file paths and model parameters.
- **dataset.py**: Handles data loading, initial cleaning, and splitting.
- **features.py**: Contains feature engineering code, such as creating new features or transforming existing features.
- **modeling/**:
  - **model_train.py**: Trains machine learning models.
  - **model_predict.py**: Runs model inference using trained models.



## Dependencies
The following Python libraries are required to run this project:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `joblib`

Install the dependencies by running:
```sh
pip install -r requirements.txt
```

## Setup Instructions
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd titanic_machine_learning
   ```
2. Create necessary directories using the `create_dirs()` function in `config.py`.
3. Place the dataset (`train.csv` and `test.csv`) into the `data/processed` directory.

## Usage
To train models and make predictions, follow these steps:

### Train Models
Run the `model_train.py` script to train the models:
```sh
python model_train.py
```
The training model is setup to choose the model with the best AUC of ROC (area under the curve of the reciever operating characteristic)

### Make Predictions
Run the `model_predict.py` script to make predictions using the trained model:
```sh
python model_predict.py
```

## Model Overview
In this project, we explored a variety of machine learning models, including:
- **Naive Bayes**: A probabilistic classifier based on Bayes' theorem.
- **Logistic Regression**: A simple linear model for binary classification.
- **Random Forest**: An ensemble method that combines multiple decision trees.
- **Voting Classifier**: An ensemble method that combines multiple models to improve accuracy.

The models were trained using 5-fold cross-validation to get reliable estimates of model performance.

The various model metrics (model_performance_metrics.txt) can be found in the `reports/performance-metrics` directory.

## Results
The Voting Classifier consistently performed the best in terms of AUC (Area Under the ROC Curve). The final submission was generated based on the predictions of the best-performing model. The model achieved an accuracy of **0.7919%** on the test data via kaggle.com.

The submission file for kaggle (`submission.csv`) can be found in the `reports/Submissions` directory.


