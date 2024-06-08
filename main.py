import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder

# Imports all training csv files
training_files = glob.glob('training_data/*.csv')

# Imports testing csv file
testing_file = glob.glob('testing_data/*.csv')[0]
testing_data = pd.read_csv(testing_file)

# Initializes an empty DataFrame for training data
training_data = pd.DataFrame()

# Reads and concatenates all training CSV files
for file in training_files:
    data = pd.read_csv(file)
    training_data = pd.concat([training_data, data], ignore_index=True)

# Defines selected columns + ignored commonly unused/irrelevant columns
selected_columns = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
    'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

# Select relevant columns from training and testing data
training_data = training_data[selected_columns]
testing_data = testing_data[selected_columns]

# Drop rows with any missing values in training data
training_data_cleaned = training_data.dropna()

# Drop rows with any missing values in testing data
testing_data_cleaned = testing_data.dropna()

# Checking for missing values in cleaned training data (should be zero if all missing values were handled)
missing_values_training_cleaned = training_data_cleaned.isnull().sum()

# Checking for missing values in cleaned testing data (should be zero if all missing values were handled)
missing_values_testing_cleaned = testing_data_cleaned.isnull().sum()

def check_missing_values(missing_values):
    for column, value in missing_values.items():
        status = "Pass" if value == 0 else value
        print(f"{column:<20}: {status}")

print("Missing values in cleaned training data:")
check_missing_values(missing_values_training_cleaned)

print("\nMissing values in cleaned testing data:")
check_missing_values(missing_values_testing_cleaned)


categorical_features = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level',
    'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc'
]

# Apply One-Hot Encoding to categorical features
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')  # drop='first' to avoid multicollinearity

# Fit and transform training data
training_encoded = pd.DataFrame(one_hot_encoder.fit_transform(training_data_cleaned[categorical_features]))
training_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_features)
training_data_cleaned = training_data_cleaned.drop(categorical_features, axis=1)
training_data_cleaned = pd.concat([training_data_cleaned.reset_index(drop=True), training_encoded.reset_index(drop=True)], axis=1)

# Transform testing data
testing_encoded = pd.DataFrame(one_hot_encoder.transform(testing_data_cleaned[categorical_features]))
testing_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_features)
testing_data_cleaned = testing_data_cleaned.drop(categorical_features, axis=1)
testing_data_cleaned = pd.concat([testing_data_cleaned.reset_index(drop=True), testing_encoded.reset_index(drop=True)], axis=1)
