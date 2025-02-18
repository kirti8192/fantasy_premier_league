import pandas as pd

# Define the base URL for the GitHub repository raw data
base_url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/refs/heads/master/data/2023-24/"

# Function to fetch and read the CSV data for any given file
def fetch_data(file_name):
    url = base_url + file_name
    try:
        # Read CSV directly from GitHub raw URL
        data = pd.read_csv(url)
        print(f"Successfully imported {file_name}")
        return data
    except Exception as e:
        print(f"Failed to fetch {file_name}: {e}")
        return None
# List to hold all the gameweek data
all_gw_data = []

# Loop through all gameweeks 1 to 38
for gw in range(1, 39):  # Gameweeks 1 to 38
    # Construct the file name dynamically for each gameweek
    file_name = f"gws/gw{gw}.csv"
    # print(file_name)
    # Fetch the data for the current gameweek
    gw_data = fetch_data(file_name)

    gw_data.to_csv("data/" + file_name[4:], index=False)
    print(f"✅ Data saved locally as {file_name[4:]}")
    
    # If data was fetched successfully, append it to the list
    if gw_data is not None:
        all_gw_data.append(gw_data)