import pandas as pd
from sklearn.model_selection import train_test_split
import config
from features import feature_engineering  # Import feature engineering functions


def load_data(train_path=config.train_file_path, test_path=config.test_file_path):
    """Load training and testing data from csv files"""
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        print(f"Train data loaded with shape: {train_data.shape}")
        print(f"Test data loaded with shape: {test_data.shape}")
        return train_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None


def init_cleaning(data):
    """perform basic cleaning, dropping cols and filling missing values."""
    columns_to_drop = ["Name", "Ticket", "Cabin"]  # modify as needed
    cleaned_data = data.drop(columns=columns_to_drop, errors="ignore")

    # print for verification
    print(f"Columns dropped. New shape: {cleaned_data.shape}")

    # Filling missing values
    if "Embarked" in cleaned_data.columns:
        cleaned_data["Embarked"] = cleaned_data["Embarked"].fillna(
            cleaned_data["Embarked"].mode()[0]
        )

    return cleaned_data


def split_data(train_data, test_size=0.3, random_state=config.RANDOM_STATE):
    # Apply feature engineering
    train_data = feature_engineering(train_data)

    # Extract only the features of interest
    feature_columns = [
        "Pclass",
        "Sex",
        "Age_band",
        "Family_category",
        "Fare",
        "Embarked",
    ]
    X = train_data[feature_columns]
    y = train_data["Survived"]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Train split shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Test split shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Example: how to use these functions by themselves
    train_data, test_data = load_data()
    if train_data is not None and test_data is not None:
        # Initial cleaning
        train_data = init_cleaning(train_data)
        test_data = init_cleaning(test_data)

        # Splitting data
        X_train, X_test, y_train, y_test = split_data(train_data)
