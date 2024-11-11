# Jakob West & Justin Landry
# 11/03/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# preprocessing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    '''Load train and test datasets.'''
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_and_show_correlation(train_path):
    '''Load the dataset, minimally preprocess for correlation analysis, and show the correlation matrix.'''
    
    # Load the train dataset
    train_data = pd.read_csv(train_path)
    
    # Minimal preprocessing to convert categorical features to numeric
    
    # Convert 'Sex' to numeric values
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    
    # Fill missing 'Age' and 'Fare' with their median values to handle NaNs
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    
    # Fill missing 'Embarked' values and one-hot encode 'Embarked' for inclusion
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)
    
    # Drop non-numeric columns like 'Name', 'Ticket', and 'Cabin' for correlation analysis
    correlation_data = train_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
    
    # Generate the correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Titanic Features')
    plt.show()

# Example usage
preprocess_and_show_correlation('titanic/train.csv')