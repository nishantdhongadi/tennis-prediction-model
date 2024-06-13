import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.impute import SimpleImputer

def load_data(file_paths):
    data = pd.DataFrame()
    for file in file_paths:
        df = pd.read_csv(file)
        data = pd.concat([data, df], ignore_index=True)
    return data

def preprocess_data(data, selected_columns):
    data = data[selected_columns]
    data = data.dropna().copy()
    return data

def check_missing_values(data):
    missing_values = data.isnull().sum()
    for column, value in missing_values.items():
        status = "Pass" if value == 0 else value
        print(f"{column:<20}: {status}")

def feature_engineering(data):
    # Ensure no leakage by only using past matches for feature calculation
    data = data.sort_values(by='tourney_date')
    data['rank_diff'] = data['winner_rank'] - data['loser_rank']
    data['age_diff'] = data['winner_age'] - data['loser_age']
    
    # Player performance metrics - calculate rolling averages for past matches
    data['winner_recent_win_pct'] = data.groupby('winner_id')['winner_rank_points'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    data['loser_recent_win_pct'] = data.groupby('loser_id')['loser_rank_points'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    
    # Head-to-Head statistics
    def h2h_wins(row):
        winner_matches = data[(data['winner_id'] == row['winner_id']) & (data['loser_id'] == row['loser_id'])]
        loser_matches = data[(data['loser_id'] == row['winner_id']) & (data['winner_id'] == row['loser_id'])]
        return len(winner_matches) - len(loser_matches)
    
    data['h2h_wins'] = data.apply(h2h_wins, axis=1)
    
    # Surface preference
    data['winner_surface_win_pct'] = data.groupby(['winner_id', 'surface'])['winner_rank_points'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    data['loser_surface_win_pct'] = data.groupby(['loser_id', 'surface'])['loser_rank_points'].transform(lambda x: x.rolling(5, min_periods=1).mean().shift(1))
    
    return data

def encode_and_scale(data, categorical_features, numerical_features, scaler, encoder, fit=True):
    if fit:
        encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    else:
        encoded_data = pd.DataFrame(encoder.transform(data[categorical_features]))
        data[numerical_features] = scaler.transform(data[numerical_features])
    encoded_data.columns = encoder.get_feature_names_out(categorical_features)
    data = data.drop(categorical_features, axis=1).reset_index(drop=True)
    data = pd.concat([data, encoded_data], axis=1)
    return data

def feature_importance(model, X_train):
    feature_names = X_train.columns
    coefficients = model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': coefficients
    })
    feature_importance_df['AbsImportance'] = feature_importance_df['Importance'].apply(abs)
    feature_importance_df = feature_importance_df.sort_values(by='AbsImportance', ascending=False)
    return feature_importance_df

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=10000, solver='newton-cg')
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {cv_scores.mean()}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Test Accuracy: {accuracy}")
    print("Classification Report on Test Data:\n", report)
    return model

# Constants
selected_columns = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
    'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

categorical_features = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level',
    'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc'
]

numerical_features = [
    'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    'rank_diff', 'age_diff'
]

columns_to_drop = [
    'winner_id', 'winner_name', 'loser_id', 'loser_name', 'tourney_date'
]

# Load data
training_files = glob.glob('training_data/*.csv')
testing_files = glob.glob('testing_data/*.csv')

training_data = load_data(training_files)
testing_data = load_data(testing_files)

# Preprocess data
training_data_cleaned = preprocess_data(training_data, selected_columns)
testing_data_cleaned = preprocess_data(testing_data, selected_columns)

# Check missing values
print("Missing values in cleaned training data:")
check_missing_values(training_data_cleaned)
print("\nMissing values in cleaned testing data:")
check_missing_values(testing_data_cleaned)

# Feature engineering
training_data_cleaned = feature_engineering(training_data_cleaned)
testing_data_cleaned = feature_engineering(testing_data_cleaned)

# Check for NaNs after feature engineering
print("Missing values in training data after feature engineering:")
check_missing_values(training_data_cleaned)
print("\nMissing values in testing data after feature engineering:")
check_missing_values(testing_data_cleaned)

# Handle NaNs by imputing or dropping (choosing to drop here)
training_data_cleaned = training_data_cleaned.dropna().copy()
testing_data_cleaned = testing_data_cleaned.dropna().copy()

# Convert 'tourney_date' to datetime format
training_data_cleaned['tourney_date'] = pd.to_datetime(training_data_cleaned['tourney_date'].astype(str), format='%Y%m%d')
testing_data_cleaned['tourney_date'] = pd.to_datetime(testing_data_cleaned['tourney_date'].astype(str), format='%Y%m%d')

# Categorical encoding and feature scaling
scaler = StandardScaler()
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

training_data_cleaned = encode_and_scale(training_data_cleaned, categorical_features, numerical_features, scaler, one_hot_encoder, fit=True)
testing_data_cleaned = encode_and_scale(testing_data_cleaned, categorical_features, numerical_features, scaler, one_hot_encoder, fit=False)

print("Training data after preprocessing:")
print(training_data_cleaned)
print("\nTesting data after preprocessing:")
print(testing_data_cleaned)

# Define the target variable
training_data_cleaned['target'] = (training_data_cleaned['winner_rank'] < training_data_cleaned['loser_rank']).astype(int)
testing_data_cleaned['target'] = (testing_data_cleaned['winner_rank'] < testing_data_cleaned['loser_rank']).astype(int)

# Drop unnecessary columns for model training
training_data_cleaned = training_data_cleaned.drop(columns_to_drop, axis=1)
testing_data_cleaned = testing_data_cleaned.drop(columns_to_drop, axis=1)

# Split into features (X) and target (y)
X_train = training_data_cleaned.drop('target', axis=1)
y_train = training_data_cleaned['target']
X_test = testing_data_cleaned.drop('target', axis=1)
y_test = testing_data_cleaned['target']

# Train and evaluate the model
model = train_and_evaluate_model(X_train, y_train, X_test, y_test)

# Feature importance analysis
feature_importance_df = feature_importance(model, X_train)
print("\nFeature Importance:")
print(feature_importance_df[['Feature', 'Importance']])
