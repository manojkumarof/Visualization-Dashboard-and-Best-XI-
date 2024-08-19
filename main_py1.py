#import csv files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

#read csv files
bat_main = pd.read_csv('batting_main.csv')
bat_avg = pd.read_csv('batting_avg.csv')
bat_sixes = pd.read_csv('batting_sixes.csv')
bat_sr = pd.read_csv('batting_sr.csv')
bowling_wickets = pd.read_csv('bowling_wickets.csv')
bowling_avg = pd.read_csv('bowling_avg.csv')
bowling_econ = pd.read_csv('bowling_econ.csv')

# Select the players only with more than 100 innings
bat_main_1 = bat_main[bat_main['Inns'] > 100]
bat_avg_1 = bat_avg[bat_avg['Inns'] > 100]
bat_sixes_1 = bat_sixes[bat_sixes['Inns'] > 100]
bat_sr_1 = bat_sr[bat_sr['Inns'] > 100]
bowling_wickets_1 = bowling_wickets[bowling_wickets['Mat'] > 100]
bowling_avg_1 = bowling_avg[bowling_avg['Mat'] > 100]
bowling_econ_1 = bowling_econ[bowling_econ['Mat'] > 100]

# There are two main dataframes, one for batting and one for bowling
# The main dataframe for batting is 'bat_main' and the main dataframe for bowling is 'bowling_wickets'
# Merge all the batting dataframes to form a new dataframe called bat_df
# Merge all the bowling dataframes to form a new dataframe called bowl_df

bat_df = bat_main_1.merge(bat_avg_1, on='Player').merge(bat_sixes_1, on='Player').merge(bat_sr_1, on='Player')
bowl_df = bowling_wickets_1.merge(bowling_avg_1, on='Player').merge(bowling_econ_1, on='Player')

bat_df = bat_df.columns.str.replace('_x', '').str.replace('_y', '')
bowl_df = bowl_df.columns.str.replace('_x', '').str.replace('_y', '')
# Define the features (X) and the target (y) for batting

# Ensure necessary columns are present in the DataFrame
# Assuming 'Avg', 'Runs', 'SR', 'Wkts', and 'Econ' are the relevant columns for the criteria

# Check if necessary columns are present
'''required_columns_bat = ['Ave', 'Runs', 'SR']
required_columns_bowl = ['Ave', 'Wkts', 'Econ']
for col in required_columns_bowl:
    if col not in bowl_df.columns:
        raise KeyError(f"Column '{col}' is not present in the dataframe.")
    
for col in required_columns_bat:
    if col not in bat_df.columns:
        raise KeyError(f"Column '{col}' is not present in the dataframe.")
'''
# Step 3: Create the `Type` column based on the criteria
def determine_type(row):


    if row['Ave'] > 25 and row['Runs'] > 3000 and row['SR'] > 120:
        return 'BATSMEN'
    elif row['Ave'] > 30 and row['Wkts'] > 30:
        return 'ALL-ROUNDERS'
    elif row['Wkts'] > 120 and row['Ave'] < 30 and row['Econ'] < 8:
        return 'BOWLERS'
    else:
        return 'UNKNOWN'  # You can set this to None or another category if needed

bat_df['Type'] = bat_df.apply(determine_type, axis=1)
bowl_df['Type'] = bowl_df.apply(determine_type, axis=1)

# Print the DataFrame with the new 'Type' column
print(bat_df[['Player', 'Ave', 'Runs', 'SR', 'Type']])
print(bowl_df[['Player', 'Ave', 'Wkts', 'Econ', 'Type']])
