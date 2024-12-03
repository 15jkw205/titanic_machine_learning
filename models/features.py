import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        # Pipeline for numerical features
        (
            "num",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median"),
                    ),  # Impute missing values for numerical features
                    ("scaler", StandardScaler()),
                ]
            ),
            [
                "Fare",
            ],
        ),
        # Pipeline for categorical features
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),  # Impute missing values for categorical features
                    (
                        "onehot",
                        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                    ),
                ]
            ),
            ["Pclass", "Age_band", "Family_category", "Sex", "Embarked"],
        ),
    ],
    remainder="passthrough",
)


# Family feature engineering
def family_feat_eng(data):
    """Add family-related features to the dataset: Family_Size and Family_category."""

    # combine parch and sibsp into 'Family_size'
    data["Family_Size"] = data["Parch"] + data["SibSp"]

    # categorize 'Family_size' into 'family_category'
    def categorize_family_size(size):
        if size == 0:
            return "no family"
        elif size <= 3:
            return "small family"
        else:
            return "large family"

    data["Family_category"] = data["Family_Size"].apply(categorize_family_size)

    # encode family_category with numbers
    family_encoder = LabelEncoder()
    data["Family_category"] = family_encoder.fit_transform(data["Family_category"])

    return data


# Age feature engineering
def age_feat_eng(data):
    """Add age-related categories to the dataset: Age_band."""

    # define bins for age categories and corresponding labels
    bins = [-float("inf"), 2, 4, 12, 18, 30, 45, 60, float("inf")]
    labels = [
        "baby",
        "infant",
        "child",
        "teenager",
        "youngadult",
        "adult",
        "oldadult",
        "elder",
    ]

    # Cut the 'Age' column into bins and add 'Age_band'
    data["Age_band"] = pd.cut(data["Age"], bins=bins, labels=labels)

    # encode Age_band with numbers
    age_band_encoder = LabelEncoder()
    data["Age_band"] = age_band_encoder.fit_transform(data["Age_band"])

    return data


# function to apply all feature engineering
def feature_engineering(data):
    """Apply all feature engineering steps to the dataset."""

    data = family_feat_eng(data)
    data = age_feat_eng(data)

    return data


# function to apply preprocessing
def preprocess_feat(X, fit=False):
    """Apply the preprocessor to the dataset.
    Parameters:
    - X: Feature dataset to process
    - fit: If True, fit and transform (use for training data)
            If False, just transform (use for test/validation data)
    """

    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    return X_transformed


if __name__ == "__main__":
    # Example usage for testing purposes
    from dataset import load_data, init_cleaning

    # Load and clean data
    train_data, test_data = load_data()
    train_data = init_cleaning(train_data)
    test_data = init_cleaning(test_data)

    # Apply feature engineering
    train_data = feature_engineering(train_data)
    test_data = feature_engineering(test_data)

    # Extract features from training data
    X_train = train_data[
        ["Pclass", "Sex", "Age_band", "Family_category", "Fare", "Embarked"]
    ]

    # Extract features from test data
    X_test = test_data[
        ["Pclass", "Sex", "Age_band", "Family_category", "Fare", "Embarked"]
    ]

    # Preprocess features
    X_train_transformed = preprocess_feat(X_train, fit=True)
    X_test_transformed = preprocess_feat(X_test)
    print(
        "Feature preprocessing complete. Transformed feature shape:",
        X_train_transformed.shape,
        X_test_transformed.shape,
    )
