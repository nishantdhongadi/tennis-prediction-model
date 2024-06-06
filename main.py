import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


# importing all csv files
csv_files = glob.glob('data/*.csv')

data_0523 = pd.DataFrame()

for file in csv_files:
    
    data = pd.read_csv(file)
    data_0523 = pd.concat([data_0523, data])

categorical_features = ['tourney_id', 'tourney_name', 'surface', 'tourney_level', 'winner_id', 'winner_name', 'winner_hand', 'winner_ioc', 'loser_id', 'loser_name', 'loser_hand', 'loser_ioc', 'score', 'round']
numerical_features = ['draw_size', 'tourney_date', 'match_num', 'winner_seed', 'winner_ht', 'winner_age', 'loser_seed', 'loser_ht', 'loser_age', 'best_of', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced', 'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)])

df = preprocessor.fit_transform(data_0523)

