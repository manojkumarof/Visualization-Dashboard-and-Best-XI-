#import csv files
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Read csv files
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

# Merging batting dataframes
bat_df = bat_main_1.merge(bat_avg_1, on='Player', how='inner', suffixes=('', '_avg')) \
                   .merge(bat_sixes_1, on='Player', how='inner', suffixes=('', '_sixes')) \
                   .merge(bat_sr_1, on='Player', how='inner', suffixes=('', '_sr'))



# Removing duplicates
bat_df.drop_duplicates(inplace=True)

# Merging bowling dataframes
bowl_df = bowling_wickets_1.merge(bowling_avg_1, on='Player', how='inner', suffixes=('', '_avg')) \
                           .merge(bowling_econ_1, on='Player', how='inner', suffixes=('', '_econ'))

# Removing duplicates
bowl_df.drop_duplicates(inplace=True)

bat_df = pd.read_csv("bat_df.csv")
bowl_df = pd.read_csv("bowl_df.csv")

# Adding hypothetical target columns
bat_df['Class'] = np.random.choice(['Type1', 'Type2'], len(bat_df))
bowl_df['Class'] = np.random.choice(['Type1', 'Type2'], len(bowl_df))

# Define the features (X) and the target (y) for batting
X_bat = bat_df[['Ave_avg', 'Runs', 'SR']]
y_bat = bat_df['Class']

# Define the features (X) and the target (y) for bowling
X_bowl = bowl_df[['Ave_avg', 'Wkts', 'Econ']]
y_bowl = bowl_df['Class']

# Split the data into training and testing sets for batting
X_bat_train, X_bat_test, y_bat_train, y_bat_test = train_test_split(X_bat, y_bat, test_size=0.2, random_state=42)

# Split the data into training and testing sets for bowling
X_bowl_train, X_bowl_test, y_bowl_train, y_bowl_test = train_test_split(X_bowl, y_bowl, test_size=0.2, random_state=42)

# Create a Random Forest Classifier for batting
rf_bat = RandomForestClassifier(n_estimators=100, random_state=42)

# Create a Random Forest Classifier for bowling
rf_bowl = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model for batting
rf_bat.fit(X_bat_train, y_bat_train)

# Fit the model for bowling
rf_bowl.fit(X_bowl_train, y_bowl_train)

# Predict the target for batting
y_bat_pred = rf_bat.predict(X_bat_test)

# Predict the target for bowling
y_bowl_pred = rf_bowl.predict(X_bowl_test)

# Print the classification report and accuracy score for batting
print("Classification Report for Batting:")
print(classification_report(y_bat_test, y_bat_pred))
print("Accuracy Score for Batting:")
print(accuracy_score(y_bat_test, y_bat_pred))

# Print the classification report and accuracy score for bowling
print("Classification Report for Bowling:")
print(classification_report(y_bowl_test, y_bowl_pred))
print("Accuracy Score for Bowling:")
print(accuracy_score(y_bowl_test, y_bowl_pred))

# Create a new dataframe with only 6 players for batting
# Create a new dataframe with only 5 players for bowling

# Select the best 6 players for batting
# find the best 2 batsmens based on the Runs
# find the best 2 batsmens based on the Ave
# find the best 2 batsmens based on the SR
# Away Players are marked as 1 and Indian players are marked as 0, Choose only 1 away player from best batsmen, there can be any number of Indian players
# Only 1 away player can be selected, all others must be Indian players


best_batsmen = bat_df.nlargest(2, 'Ave').append(bat_df.nlargest(2, 'Runs')).append(bat_df[bat_df['Away']== 0 ].nlargest(2, 'SR'))

# Select the best 5 players for bowling
# find the best 2 bowlers based on the Wkts
# find the best 2 bowlers based on the Econ
# find the best bowler based on the Ave


best_bowlers = bowl_df.nlargest(2, 'Wkts').append(bowl_df.nsmallest(2, 'Econ')).append(bowl_df.nsmallest(1, 'Ave_avg')).append(bowl_df.nsmallest(1, 'SR')).append(bowl_df.nlargest(1, '5_econ'))
# 

# Remove duplicates if any, and replace the duplicate values with the next best player
best_bowlers.drop_duplicates(inplace=True)
best_bowlers = best_bowlers.head(5)


# Concatenate the best batsmen and best bowlers dataframes
best_players = pd.concat([best_batsmen, best_bowlers])

# Display the best players
print("Best Playing XI")
print(best_players[['Player']])

# Save the best players to a CSV file
best_players[['Player']].to_excel('best_players.xlsx')
