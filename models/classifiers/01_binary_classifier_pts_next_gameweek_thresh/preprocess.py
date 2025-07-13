import pandas as pd
import sys, os

# read the data
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw'))
csv_path = os.path.join(base_dir, 'vaastav_2022_23.csv')
df = pd.read_csv(csv_path)

# select features
cols_to_keep = ['minutes',
                'team',
                'goals_scored', 
                'assists', 
                'expected_goals', 
                'expected_assists', 
                'clean_sheets',
                'ict_index',
                'bps', 
                'bonus', 
                'total_points',
                ]

# pivot table to get unified dataframe
df_multigw = df.pivot_table(index='element', 
                columns = 'GW', 
                values = cols_to_keep, 
                aggfunc='sum').reset_index()

# rename columns to include gameweek number
df_multigw.columns = [f"{col}_gw{int(gw)}" if isinstance(gw, (int,float)) else col for col,gw in df_multigw.columns]

# write a function that drops all data for gameweek that is targeted
def get_df_for_gw(gw):
    """
    Returns a DataFrame with all data for the gameweek just before the one that is targeted.
    """
    if gw == 1:
        raise ValueError("Gameweek 1 does not have a previous gameweek to reference.")
    if gw > 38:
        raise ValueError("Gameweek must be between 1 and 38.")
    
    # get the columns for the gameweek just before the one that is targeted
    pass