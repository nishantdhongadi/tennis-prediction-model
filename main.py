import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# importing all csv files
csv_files = glob.glob('training_data/*.csv')
testing_files = glob.glob('testing_data/*.csv')
testing_files = pd.read_csv(testing_files)

training_data = pd.DataFrame()

for file in csv_files:
    
    data = pd.read_csv(file)
    training_data = pd.concat([training_data, data])

selected_columns = [
    'tourney_id', 'tourney_name', 'surface', 'tourney_level', 'tourney_date',
    'winner_id', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age', 'winner_rank', 'winner_rank_points',
    'loser_id', 'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
    'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
    'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced'
]

training_data = training_data[selected_columns]

missing_values = training_data.isnull().sum()

print(missing_values)