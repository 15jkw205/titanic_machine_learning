# Jakob West & Justin Landry
# 11/02/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# based on certain features like age, gender, ticket, class, etc...
# models.py

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import catboost as cb


# Linear Regression for Classification
def train_logistic_regression(x_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=0, C=0.5, solver='liblinear')
    model.fit(x_train, y_train)
    return model


# Tree-Based Models
def train_decision_tree(x_train, y_train):
    model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=10, min_samples_leaf=5)
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_split=10, min_samples_leaf=4, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_extra_trees(x_train, y_train):
    model = ExtraTreeClassifier(max_depth=7, min_samples_split=5, min_samples_leaf=3, random_state=0)
    model.fit(x_train, y_train)
    return model

# Gradient Boosting (special tree models)
def train_gradient_boosting(x_train, y_train):
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, min_samples_split=8, min_samples_leaf=3, random_state=0)
    model.fit(x_train, y_train)
    return model

def train_sgd_classifier(x_train, y_train):
    model = SGDClassifier(max_iter=1000, tol=1e-3, alpha=0.0001, learning_rate='optimal', random_state=0)
    model.fit(x_train, y_train)
    return model

def train_hist_gradient_boosting(x_train, y_train):
    model = HistGradientBoostingClassifier(random_state=0)
    model.fit(x_train, y_train)
    return model


# Ensemble Methods
def train_adaboost(x_train, y_train):
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.6, random_state=0, algorithm='SAMME')
    model.fit(x_train, y_train)
    return model

def train_xgboost(x_train, y_train):
    model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=0, eval_metric='mlogloss')
    model.fit(x_train, y_train)
    return model

def train_catboost(x_train, y_train):
    model = cb.CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_state=0, verbose=0)
    model.fit(x_train, y_train)
    return model


# Support Vector Machine (SVM)
def train_svm(x_train, y_train):
    model = SVC(C=1.5, kernel='rbf', gamma=0.1, random_state=0)
    model.fit(x_train, y_train)
    return model


# Nearest Neighbor Models
def train_knn(x_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)
    model.fit(x_train, y_train)
    return model

# Naive Bayes Models
def train_gaussian_nb(x_train, y_train):
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(x_train, y_train)
    return model

def train_multinomial_nb(x_train, y_train):
    model = MultinomialNB(alpha=0.1)
    model.fit(x_train, y_train)
    return model

def train_bernoulli_nb(x_train, y_train):
    model = BernoulliNB(alpha=0.5)
    model.fit(x_train, y_train)
    return model


# Neural Network
def train_ann(x_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, alpha=0.0001, learning_rate='adaptive', random_state=0)
    model.fit(x_train, y_train)
    return model


# Dimensionality Reduction (used as preprocessing step)
def train_pca(x_train, y_train):
    model = PCA(n_components=2, random_state=0)
    model.fit(x_train)
    return model

def train_lda(x_train, y_train):
    model = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    model.fit(x_train, y_train)
    return model

def train_qda(x_train, y_train):
    model = QuadraticDiscriminantAnalysis(reg_param=0.1)
    model.fit(x_train, y_train)
    return model


# Powerful classifiers
def train_bagging_classifier(x_train, y_train):
    model = BaggingClassifier(
        n_estimators=100,
        random_state=0
    )
    model.fit(x_train, y_train)
    return model

def train_voting_classifier(x_train, y_train):
    model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)),
            ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=0)),
            ('cat', cb.CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_state=0, verbose=0))
        ],
        voting='soft'
    )
    model.fit(x_train, y_train)
    return model

def train_stacking_classifier(x_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=7, random_state=0)),
        ('xgb', xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=4, random_state=0)),
        ('cat', cb.CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, random_state=0, verbose=0))
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=0)
    )
    model.fit(x_train, y_train)
    return model


# Evaluation Function
def evaluate_model(model, x_val, y_val):
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def split_data(x, y, test_size=0.2, random_state=0):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


# Function to train and evaluate all models
def train_all_models(x, y):
    x_train, x_val, y_train, y_val = split_data(x, y)
    accuracy_results = {}

    # Training and evaluating each model
    models = {
        'Logistic Regression': train_logistic_regression,
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest,
        'Extra Trees': train_extra_trees,
        'Gradient Boosting': train_gradient_boosting,
        'Stochastic Gradient Descent': train_sgd_classifier,
        'AdaBoost': train_adaboost,
        'XGBoost': train_xgboost,
        'CatBoost': train_catboost,
        'SVM': train_svm,
        'K-Nearest Neighbors': train_knn,
        'Gaussian NB': train_gaussian_nb,
        'Multinomial NB': train_multinomial_nb,
        'Bernoulli NB': train_bernoulli_nb,
        'Artificial Neural Network': train_ann,
        'Bagging Classifier': train_bagging_classifier,
        'LDA': train_lda,
        'QDA': train_qda,
        'Voting Classifier': train_voting_classifier,
        'Hist Gradient Boosting': train_hist_gradient_boosting,
        'Stacking Classifier': train_stacking_classifier
    }

    # Evaluate each model and store accuracy
    for model_name, train_function in models.items():
        model = train_function(x_train, y_train)
        accuracy = evaluate_model(model, x_val, y_val)
        accuracy_results[model_name] = accuracy

    return accuracy_results