# fetch_data.py
import os
import requests, pandas as pd

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    data = requests.get(url).json()
    players_df = pd.DataFrame(data['elements'])
    os.makedirs("data/raw", exist_ok=True)
    players_df.to_csv("data/raw/bootstrap_players.csv", index=False)

if __name__=='__main__':
    fetch_bootstrap()