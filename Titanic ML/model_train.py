from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    roc_auc_score,
)
import joblib  # For saving models
import config
from dataset import load_data, init_cleaning, split_data
from features import feature_engineering, preprocessor


def train_models(X_train, y_train):
    # Initialize a variable to store results, average metrics, and best model tracking
    results = {}
    training_metrics = []
    best_model_name = None
    best_model_auc = 0

    # Use stratified Kfold for cross-validation
    kfold = StratifiedKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_STATE
    )

    # Loop through each model in your models dictionary
    for name, model in config.MODELS.items():
        # Create a pipeline with preprocessor and classifier
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", model)]
        )

        # Perform cross-validation predictions
        y_train_pred = cross_val_predict(
            pipeline, X_train, y_train, cv=kfold, method="predict"
        )

        # Calculate metrics using cross-validation predictions
        accuracy = accuracy_score(y_train, y_train_pred)
        precision = precision_score(y_train, y_train_pred, average="binary")
        recall = recall_score(y_train, y_train_pred, average="binary")
        f1 = f1_score(y_train, y_train_pred, average="binary")

        mse = mean_squared_error(y_train, y_train_pred)

        # Calculate AUC of ROC
        try:
            y_train_proba = cross_val_predict(
                pipeline, X_train, y_train, cv=kfold, method="predict_proba"
            )[:, 1]
            roc_auc = roc_auc_score(y_train, y_train_proba)
        except AttributeError:
            # Some models don't have predict_proba (e.g., SVM without probability=True)
            y_train_decision = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=kfold,
                method="decision_function",
            )
            roc_auc = roc_auc_score(y_train, y_train_decision)

        # Store the average metrics
        metrics = f"{name} (Averaged over 5-Fold Cross Validation):\n"
        metrics += f"Accuracy: {accuracy:.4f}\n"
        metrics += f"Precision: {precision:.4f}\n"
        metrics += f"Recall: {recall:.4f}\n"
        metrics += f"F1 Score: {f1:.4f}\n"
        metrics += f"AUC of ROC: {roc_auc:.4f}\n"
        metrics += f"mean squared error: {mse:.4f}\n"
        # metrics += f"mean absolute error: {mae:.4f}\n"
        print(metrics)

        # Append metrics to list
        training_metrics.append(metrics)

        # Update best model based on AUC of ROC
        if roc_auc > best_model_auc:
            best_model_auc = roc_auc
            best_model_name = name
            # Store the best performing pipeline for validation
            results[name] = pipeline

    # Save the best model using joblib
    if best_model_name:
        best_pipeline = results[best_model_name]

        # fit the best model on the entire training dataset to ensure fully fit preprocessor
        best_pipeline.fit(X_train, y_train)

        model_file_path = config.REPORTS_DIR / "best_model.pkl"
        joblib.dump(best_pipeline, model_file_path)
        print(f"Best model '{best_model_name}' saved to: {model_file_path}")

    # Print the model with the highest AUC of ROC
    print(
        f"\nModel with the highest AUC of ROC: {best_model_name} ({best_model_auc:.4f})\n"
    )

    # Get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_file_path = config.metrics_file_path
    with open(metrics_file_path, "w") as f:
        f.write(f"Training Metrics Report - Generated on {current_datetime}\n")
        f.write("=" * 50 + "\n\n")
        for metric in training_metrics:
            f.write(metric + "\n")
    print(f"Performance metrics saved to: {metrics_file_path}")


if __name__ == "__main__":
    # Load and prepare data
    train_data, _ = load_data()
    if train_data is not None:
        # initial cleaning and feature engineering
        train_data = init_cleaning(train_data)
        train_data = feature_engineering(train_data)

        # split the data into training and test splits
        X_train, X_test, y_train, y_test = split_data(train_data)

        # Train models
        train_models(X_train, y_train)
