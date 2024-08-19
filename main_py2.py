import pandas as pd

# Read CSV files
bat_main = pd.read_csv('batting_main.csv')
bat_avg = pd.read_csv('batting_avg.csv')
bat_sixes = pd.read_csv('batting_sixes.csv')
bat_sr = pd.read_csv('batting_sr.csv')

# Select the players only with more than 100 innings
bat_main_1 = bat_main[bat_main['Inns'] > 100]
bat_avg_1 = bat_avg[bat_avg['Inns'] > 100]
bat_sixes_1 = bat_sixes[bat_sixes['Inns'] > 100]
bat_sr_1 = bat_sr[bat_sr['Inns'] > 100]

# Merge the dataframes on a common key, assuming 'Player' is a common column
df = bat_main_1.merge(bat_avg_1, on='Player').merge(bat_sixes_1, on='Player').merge(bat_sr_1, on='Player')

# Rename columns to remove suffixes
df.columns = df.columns.str.replace('_x', '').str.replace('_y', '')

# Rename 'Avg' to 'Ave'
df = df.rename(columns={'Avg': 'Ave'})

# Print merged dataframe columns to verify
print("Merged dataframe columns:", df.columns)

# Ensure necessary columns are present in the DataFrame
# Assuming 'Ave', 'Runs', 'SR', 'Wkts', and 'Econ' are the relevant columns for the criteria

# Check if necessary columns are present
required_columns = ['Ave', 'Runs', 'SR', 'Wkts', 'Econ']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' is not present in the dataframe.")

# Define the criteria for each type of player
def determine_type(row):
    if row['Ave'] > 30 and row['Runs'] > 3000 and row['SR'] > 120:
        return 'BATSMEN'
    elif row['Ave'] > 30 and row['Wkts'] > 30:
        return 'ALL-ROUNDERS'
    elif row['Wkts'] > 120 and row['Ave'] < 30 and row['Econ'] < 8:
        return 'BOWLERS'
    else:
        return 'UNKNOWN'  # You can set this to None or another category if needed
# Create the 'Type' column based on the criteria
df['Type'] = df.apply(determine_type, axis=1)

# Print the DataFrame with the new 'Type' column
print(df[['Player', 'Ave', 'Runs', 'SR', 'Wkts', 'Econ', 'Type']])
