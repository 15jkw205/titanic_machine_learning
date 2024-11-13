# Jakob West & Justin Landry
# 11/02/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# main.py

from preprocessing import preprocess_data
from models import train_all_models

# Paths to your train and test datasets
train_path = 'titanic/train.csv'
test_path = 'titanic/test.csv'

# Preprocess the training data
train_data, _ = preprocess_data(train_path, test_path)

# Select the standard features and target for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
x = train_data[features]
y = train_data['Survived']

# Train and evaluate all models
model_accuracies = train_all_models(x, y)

# Print model performance
print("\nModel Performance on Training Data:")
for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: {accuracy:.4f}")