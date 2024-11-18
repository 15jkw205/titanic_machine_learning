from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Base directory: starting point for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# file paths
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "Performance-metrics"


train_file_path = DATA_DIR / "train.csv"
test_file_path = DATA_DIR / "test.csv"
metrics_file_path = REPORTS_DIR / "performance-metrics/model_performance_metrics.txt"
submission_file_path = REPORTS_DIR / "lightGBM_classifier.csv"


# function to make sure directories exist
def create_dirs():
    """Create necessary directories if they dont't exist"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# random seed
RANDOM_STATE = 42

# cross validation number
N_SPLITS = 5

MODELS = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "K-Nearest Neighbours": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=0),
    # "Bagging Decision Tree": BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
    # "Boosted Decision Tree": AdaBoostClassifier(n_estimators=50, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Voting Classifier": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("knn", KNeighborsClassifier()),
            ("svc", SVC(probability=True)),
            # ("XGBoost", xgb.XGBClassifier(n_estimators=100, random_state=0)),
            # ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42))
            # ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=0))
        ],
        voting="soft",
    ),
    "Neural Network": MLPClassifier(max_iter=1000),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=150,  # Number of boosting rounds (trees)
        learning_rate=0.05,  # Step size at each iteration
        max_depth=-1,  # Maximum depth of a tree (-1 means no limit)
        random_state=0,  # For reproducibility
        boosting_type="gbdt",  # Gradient Boosting Decision Tree
        subsample=0.8,  # Fraction of data used to build each tree
        colsample_bytree=0.8,  # Fraction of features used in building each tree
    ),
}
