import pandas as pd
import glob
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score


''' Importing and Preprocessing Data '''

training_files = glob.glob('training_data/*.csv')
testing_files = glob.glob('testing_data/*.csv')              

training_data = pd.DataFrame()                              
testing_data = pd.DataFrame()                                  

for file in training_files:                                 
    data = pd.read_csv(file)
    training_data = pd.concat([training_data, data], ignore_index=True)
    
    
for file in testing_files:                                  
    data = pd.read_csv(file)
    testing_data = pd.concat([testing_data, data], ignore_index=True)

# Defines selected columns + ignored commonly unused/irrelevant columns
selected_columns = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
    'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

training_data = training_data[selected_columns]
testing_data = testing_data[selected_columns]

# Drops rows with any missing values in training/testing data
training_data_cleaned = training_data.dropna().copy()
testing_data_cleaned = testing_data.dropna().copy()

# Checks for missing values in cleaned training/testing data (should be zero if all missing values were handled)
missing_values_training_cleaned = training_data_cleaned.isnull().sum()
missing_values_testing_cleaned = testing_data_cleaned.isnull().sum()

def check_missing_values(missing_values):
    for column, value in missing_values.items():
        status = "Pass" if value == 0 else value
        print(f"{column:<20}: {status}")

print("Missing values in cleaned training data:")
check_missing_values(missing_values_training_cleaned)

print("\nMissing values in cleaned testing data:")
check_missing_values(missing_values_testing_cleaned)

training_data_cleaned['tourney_date'] = pd.to_datetime(training_data_cleaned['tourney_date'].astype(str), format='%Y%m%d')      # Convert 'tourney_date' to datetime format
testing_data_cleaned['tourney_date'] = pd.to_datetime(testing_data_cleaned['tourney_date'].astype(str), format='%Y%m%d')



''' Feature Engineering '''

training_data_cleaned['rank_diff'] = training_data_cleaned['winner_rank'] - training_data_cleaned['loser_rank']
training_data_cleaned['age_diff'] = training_data_cleaned['winner_age'] - training_data_cleaned['loser_age']

testing_data_cleaned['rank_diff'] = testing_data_cleaned['winner_rank'] - testing_data_cleaned['loser_rank']
testing_data_cleaned['age_diff'] = testing_data_cleaned['winner_age'] - testing_data_cleaned['loser_age']


''' Categorical Encoding and Feature Scaling'''

categorical_features = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level',
    'winner_hand', 'winner_ioc', 'loser_hand', 'loser_ioc'
]

# Using One-Hot Encoding to categorical features - 
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore') 

# Fit and transform training data
training_encoded = pd.DataFrame(one_hot_encoder.fit_transform(training_data_cleaned[categorical_features]))
training_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_features)
training_data_cleaned = training_data_cleaned.drop(categorical_features, axis=1).reset_index(drop=True)
training_data_cleaned = pd.concat([training_data_cleaned, training_encoded], axis=1)

# Transform testing data
testing_encoded = pd.DataFrame(one_hot_encoder.transform(testing_data_cleaned[categorical_features]))
testing_encoded.columns = one_hot_encoder.get_feature_names_out(categorical_features)
testing_data_cleaned = testing_data_cleaned.drop(categorical_features, axis=1).reset_index(drop=True)
testing_data_cleaned = pd.concat([testing_data_cleaned, testing_encoded], axis=1)

numerical_features = [
    'winner_ht', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_ht', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
    'rank_diff', 'age_diff'
]

scaler = StandardScaler()

training_data_cleaned[numerical_features] = scaler.fit_transform(training_data_cleaned[numerical_features])
testing_data_cleaned[numerical_features] = scaler.transform(testing_data_cleaned[numerical_features])

print("Training data after preprocessing:")
print(training_data_cleaned)

print("\nTesting data after preprocessing:")
print(testing_data_cleaned)

''' Feature Importance Analysis '''

feature_names = X_train.columns
coefficients = model.coef_[0]

# Create a DataFrame to display feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': coefficients
})

# Sort by absolute value of importance
feature_importance['AbsImportance'] = feature_importance['Importance'].apply(abs)
feature_importance = feature_importance.sort_values(by='AbsImportance', ascending=False)

print("\nFeature Importance:")
print(feature_importance[['Feature', 'Importance']])


''' Model Training and Testing '''

# Define the target variable
training_data_cleaned['target'] = (training_data_cleaned['winner_rank'] < training_data_cleaned['loser_rank']).astype(int)
testing_data_cleaned['target'] = (testing_data_cleaned['winner_rank'] < testing_data_cleaned['loser_rank']).astype(int)


# Drop unnecessary columns for model training
columns_to_drop = [
    'winner_id', 'winner_name', 'loser_id', 'loser_name', 'tourney_date'
]
training_data_cleaned = training_data_cleaned.drop(columns_to_drop, axis=1)
testing_data_cleaned = testing_data_cleaned.drop(columns_to_drop, axis=1)


# Split into features (X) and target (y)
X_train = training_data_cleaned.drop('target', axis=1)
y_train = training_data_cleaned['target']
X_test = testing_data_cleaned.drop('target', axis=1)
y_test = testing_data_cleaned['target']

# Initialize the model
model = LogisticRegression(max_iter=1000)


''' Cross-Validation and Final Evaluation '''

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# Train the model on the entire training data and evaluate on the test set
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Test Accuracy: {accuracy}")
print("Classification Report on Test Data:\n", report)



