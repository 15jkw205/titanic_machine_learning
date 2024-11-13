# Jakob West & Justin Landry
# 11/11/2024
# Titanic Machine Learning
# CS 3820-001 - Introduction to Artificial Intelligence
# Problem: To predict whether a passenger on the Titanic survived or not
# preprocessing.py


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

def load_data(train_path):
    '''Load train dataset.'''
    return pd.read_csv(train_path)

def extract_title(df):
    '''Extract title (Mr, Mrs, Miss, Master, etc.) from the 'Name' column.'''
    df['Title'] = df['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
    
    # Correct handling of titles, including 'Master'
    rare_titles = ['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Jonkheer', 'Don', 'Sir', 'Lady', 'Countess', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    
    # Explicitly retain 'Master'
    title_list = ['Mr', 'Mrs', 'Miss', 'Master', 'Rare']
    for title in title_list:
        df[title] = (df['Title'] == title).astype(int)
    
    return df

def create_family_features(df):
    '''Create family size and related features.'''
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    return df

def extract_deck(df):
    '''Extract deck information from the 'Cabin' feature.'''
    df['Deck'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')
    df = pd.get_dummies(df, columns=['Deck'], drop_first=False)
    return df

def process_ticket(df):
    '''Create features from ticket information.'''
    df['TicketPrefix'] = df['Ticket'].apply(lambda x: x.split()[0] if len(x.split()) > 1 else 'None')
    df['TicketGroupSize'] = df.groupby('Ticket')['Ticket'].transform('count')
    df = pd.get_dummies(df, columns=['TicketPrefix'], drop_first=False)
    return df

def age_binning(df):
    '''Bin the 'Age' feature into categories.'''
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior', 'Elder'])
    df = pd.get_dummies(df, columns=['AgeBin'], drop_first=False)
    return df

def fare_binning(df):
    '''Bin the 'Fare' feature into categories.'''
    df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
    df = pd.get_dummies(df, columns=['FareBin'], drop_first=False)
    return df

def scale_features(df, features):
    '''Scale numerical features using Min-Max scaling.'''
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    return df

def polynomial_features(df, features):
    '''Generate polynomial interaction features.'''
    df[features] = df[features].fillna(0)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    poly_features = poly.fit_transform(df[features])
    poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(features))
    df = pd.concat([df, poly_df], axis=1)
    return df

def feature_selection(df, original_features):
    '''Perform feature selection and retain critical features.'''
    df_numeric = df.select_dtypes(include=[np.number])
    selector = VarianceThreshold(threshold=0.01)
    df_selected = selector.fit_transform(df_numeric)
    selected_columns = df_numeric.columns[selector.get_support()]
    
    df = pd.DataFrame(df_selected, columns=selected_columns)
    
    # Ensure critical features like 'Age' are retained
    for feature in original_features:
        if feature not in df.columns:
            df[feature] = df_numeric[feature]
    
    return df

def advanced_preprocessing(df):
    '''Apply all feature engineering and preprocessing steps.'''
    original_features = ['Age']  # Ensure 'Age' is preserved
    
    df = extract_title(df)
    df = create_family_features(df)
    df = extract_deck(df)
    df = process_ticket(df)
    df = age_binning(df)
    df = fare_binning(df)
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    df = scale_features(df, ['Age', 'Fare', 'FamilySize'])
    df = polynomial_features(df, ['Age', 'Fare'])
    
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
    df = feature_selection(df, original_features)
    
    return df

def generate_correlation_matrix(train_path):
    '''Generate and display the correlation matrix.'''
    train_data = load_data(train_path)
    train_data = advanced_preprocessing(train_data)
    
    correlation_matrix = train_data.corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix After Feature Engineering')
    plt.show()

# Generate the correlation matrix
# generate_correlation_matrix('titanic/train.csv')