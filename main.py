# Jakob West & Justin Landry
# 11/12/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# main.py


from preprocessing import advanced_preprocessing
from models import run_all_models
import pandas as pd
from sklearn.model_selection import train_test_split

train_path = 'titanic/train.csv'
validation_path = 'titanic/validation.csv'

# Preprocess the training data
print("Preprocessing the training data...")
train_data = advanced_preprocessing(pd.read_csv(train_path))
features = train_data.drop(columns='Survived').columns.tolist()
x = train_data[features]
y = train_data['Survived']

# Split data into training and testing sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

def run_all():
    """Run all models, evaluate performance, and select the best model."""
    print("\nRunning All Models with 80/20 split...")
    model_accuracies, best_model_name, best_model = run_all_models(x_train, x_val, y_train, y_val)
    
    # Display the performance of all models
    print("\nAll Model Performance:")
    for model_name, accuracy in model_accuracies.items():
        print(f"{model_name}: {accuracy:.4f}")
    
    print(f"\nBest Model Selected: {best_model_name} with accuracy: {model_accuracies[best_model_name]:.4f}")
    return best_model

def generate_submission(best_model):
    """Generate predictions for Kaggle submission using the best model."""
    print("\nGenerating Kaggle submission...")
    validation_data = advanced_preprocessing(pd.read_csv(validation_path))
    x_val = validation_data[features]

    # Generate predictions using the best model
    predictions = best_model.predict(x_val)

    # Create a submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': validation_data['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file 'submission.csv' created successfully!")

# --- MAIN EXECUTION FLOW ---
best_model = run_all()
# Uncomment the line below to generate the submission
# generate_submission(best_model)