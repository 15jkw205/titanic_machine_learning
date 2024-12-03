import pandas as pd
import joblib
import config
from dataset import load_data, init_cleaning
from features import feature_engineering


def predict():
    # Load the saved best model
    try:
        model_file_path = config.REPORTS_DIR / "best_model.pkl"
        chosen_model = joblib.load(model_file_path)
        print(f"loaded model from: {model_file_path}")
    except FileNotFoundError as e:
        print(f"Error: Model file not found. {e}")
        return

    # Load validation data
    _, validation_data = load_data()
    if validation_data is None:
        print("Validation data not found.")
        return

    # Initial cleaning of validation data
    validation_data = init_cleaning(validation_data)

    # Apply feature engineering
    validation_data = feature_engineering(validation_data)

    # Extract features from validation data
    X_validation = validation_data[
        ["Pclass", "Sex", "Age_band", "Family_category", "Fare", "Embarked"]
    ]

    # predict using chosen model
    y_validation_prediction = chosen_model.predict(X_validation)

    submission = pd.DataFrame(
        {
            "PassengerId": validation_data["PassengerId"],
            "Survived": y_validation_prediction,
        }
    )

    # Save the submission to a CSV file
    submission_file_path = config.SUBMISSION_DIR / "submission.csv"
    submission.to_csv(submission_file_path, index=False)
    print(f"Submission file saved to: {submission_file_path}")


if __name__ == "__main__":
    predict()
