# fetch_data.py
import os
import requests, pandas as pd

def fetch_bootstrap():
    """
    Fetches the Fantasy Premier League (FPL) bootstrap-static JSON data from the official API.
    Extracts the 'elements' table containing current season player information and saves it
    as a CSV file at 'data/raw/bootstrap_players.csv'.

    The dataset includes aggregated season statistics such as total points, cost, form, and more.
    """
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    players_df = pd.DataFrame(data['elements'])
    os.makedirs("data/raw", exist_ok=True)
    players_df.to_csv("data/raw/bootstrap_players.csv", index=False)

def fetch_github_data_vaastav():
    """
    Downloads and saves historical Fantasy Premier League (FPL) gameweek-level data
    from the Vaastav dataset hosted on GitHub to 'data/raw/vaastav_2022_23.csv'.

    Returns:
        pd.DataFrame: DataFrame containing player gameweek statistics for the 2022–23 season.
    """

    # get data from Vaastav's GitHub repository
    url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2022-23/gws/merged_gw.csv"
    os.makedirs("data/raw", exist_ok=True)
    filepath = "data/raw/vaastav_2022_23.csv"
    df = pd.read_csv(url)

    # ensure the directory exists
    os.makedirs("data/raw", exist_ok=True)

    # save the DataFrame to a CSV file
    df.to_csv(filepath, index=False)

if __name__=='__main__':
    fetch_github_data_vaastav()