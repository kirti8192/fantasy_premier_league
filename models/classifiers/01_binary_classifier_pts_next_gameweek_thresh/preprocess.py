import pandas as pd
import os
import re

def get_target_column(df_this, gw):
    """
    Returns the boolean 
    """
    target_col = f'total_points_gw{gw}'
    return df_this[target_col] > 4  # threshold for next gameweek points to be considered as a good performance

def get_df_for_gw(df_this, gw):
    """
    Returns a DataFrame with all data for the gameweek just before the one that is targeted.
    """
    if gw == 1:
        raise ValueError("Gameweek 1 does not have a previous gameweek to reference.")
    if gw > 38:
        raise ValueError("Gameweek must be between 1 and 38.")
    
    # get the columns for the gameweek just before the one that is targeted
    static_cols_to_keep = [col for col in df_this.columns if "_gw" not in col]
    gw_suffixes = [f"_gw{idx}" for idx in range(1, gw)]
    gw_cols_to_keep = [col for col in df_this.columns for suffix in gw_suffixes if col.endswith(suffix) ]

    # Filter the DataFrame to keep only the desired columns
    cols_to_keep = static_cols_to_keep + gw_cols_to_keep
    df_filtered = df_this[cols_to_keep]

    # get target column
    df_target_col = get_target_column(df_this, gw)

    # merge df_filtered with the target column
    df_filtered = df_filtered.merge(df_target_col.rename('target'), left_index=True, right_index=True)

    return df_filtered

def get_df():
    """
    Returns a DataFrame with all data for the 2022–23 Fantasy Premier League season.
    The DataFrame contains aggregated statistics for each player across all gameweeks.
    """

    # read the data
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../data/raw'))
    csv_path = os.path.join(base_dir, 'vaastav_2022_23.csv')
    df = pd.read_csv(csv_path)

    # extract team and position information
    df_common = df.groupby('element').agg({'team': 'first',
                                             'position': 'first'
                                             }).reset_index()

    # select features
    cols_to_keep = ['minutes',
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

    # flatten columns with gameweek suffix
    df_multigw.columns = [f"{col}_gw{int(gw)}" if isinstance(gw, (int,float)) else col for col,gw in df_multigw.columns]

    # merge common information with gamew data
    df_multigw = df_common.merge(df_multigw, on='element', how='left')

    # set element as the index
    df_multigw.set_index('element', inplace=True)

    return df_multigw

if __name__ == '__main__':

    df = get_df()
    print(df.head())

    # extract just data to predict gw X
    gw = 10
    df_gw = get_df_for_gw(df, gw)
    print(df_gw.shape)