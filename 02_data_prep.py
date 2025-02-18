import pandas as pd

# parse all gw data from csv
df_list = []
for gw in range(1,39):
    df = pd.read_csv(f'data/gw{gw}.csv')
    df_list.append(df)