# Jakob West & Justin Landry
# 11/11/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# simple_preprocessing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    '''Load train and test datasets.'''
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def basic_preprocessing(train_data):
    '''Basic preprocessing: handle missing values and convert categorical features to numerical.'''
    # Convert 'Sex' to numeric
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
    
    # Handle missing values for 'Age' and 'Fare'
    train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
    train_data['Fare'] = train_data['Fare'].fillna(train_data['Fare'].median())
    
    # Fill missing 'Embarked' values
    train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
    
    # One-hot encode 'Embarked' without dropping any column
    train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=False)
    
    return train_data

def preprocess_data(train_path, test_path):
    '''Preprocess the training and test data using basic preprocessing.'''
    train_data, test_data = load_data(train_path, test_path)
    train_data = basic_preprocessing(train_data)
    test_data = basic_preprocessing(test_data)
    return train_data, test_data

def generate_correlation_matrix(train_path):
    '''Generate and display the correlation matrix for the given dataset.'''
    
    # Load the train dataset
    train_data = pd.read_csv(train_path)
    
    # Perform basic preprocessing for correlation analysis
    train_data = basic_preprocessing(train_data)
    
    # Drop non-numeric columns for correlation analysis
    correlation_data = train_data.drop(['Cabin', 'Name', 'Ticket', 'PassengerId'], axis=1)
    
    # Generate the correlation matrix
    correlation_matrix = correlation_data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=(14, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Titanic Features with All Embarked Categories')
    plt.show()

# Uncomment the line below to generate the correlation matrix when needed
generate_correlation_matrix('titanic/train.csv')