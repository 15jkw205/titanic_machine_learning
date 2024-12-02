from pathlib import Path
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Base directory: starting point for relative paths
BASE_DIR = Path(__file__).resolve().parent.parent

# file paths
DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "performance-metrics"
SUBMISSION_DIR = BASE_DIR / "reports" / "Submissions"


train_file_path = DATA_DIR / "train.csv"
test_file_path = DATA_DIR / "test.csv"
metrics_file_path = REPORTS_DIR / "performance-metrics"
submission_file_path = SUBMISSION_DIR


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
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Voting Classifier": VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("knn", KNeighborsClassifier()),
            ("svc", SVC(probability=True)),
        ],
        voting="soft",
    ),
    "Neural Network": MLPClassifier(max_iter=1000),
}
