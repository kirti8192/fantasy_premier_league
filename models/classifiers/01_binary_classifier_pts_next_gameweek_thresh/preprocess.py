import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from tensorflow import keras 
from tensorflow.keras import layers, models


def preprocess_values(df_this):
    """
    This function handles NaN values and converts object columns to numeric types.
    """

    # handle NaN values
    df_this = df_this.fillna(0)  # Replace NaNs with 0s
    
    # handle categorical values
    categorical_cols = list(df_this.select_dtypes(include=['object']).columns)

    # one-hot encode categorical columns
    myOneHotEncoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_encoded = pd.DataFrame(myOneHotEncoder.fit_transform(df_this[categorical_cols]))

    # Set the column names to the one-hot encoded feature names
    df_encoded.columns = myOneHotEncoder.get_feature_names_out(categorical_cols)
    df_encoded.index = df_this.index  # Ensure the index matches the original DataFrame

    # drop original categorical columns
    df_this = df_this.drop(columns=categorical_cols, axis=1)

    # concatenate the one-hot encoded DataFrame with the original DataFrame
    df_concat = pd.concat([df_encoded, df_this], axis=1)

    return df_concat

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
                    # 'expected_goals',     # all zeros
                    # 'expected_assists',   # all zeros
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

    # extract just data to predict gw X
    gw = 5
    df_gw = get_df_for_gw(df, gw)

    # value preprocess
    df_gw = preprocess_values(df_gw)

    # extract X and y
    y = df_gw.target
    X = df_gw.drop(columns=['target'], axis=1)

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # random forest binary classifier
    forest_model = RandomForestClassifier(n_estimators=100, random_state=0)
    forest_model.fit(X_train, y_train)
    print("Model trained successfully.")

    forest_prediction = forest_model.predict(X_test)
    print("Predictions made successfully.")

    forest_accuracy = forest_model.score(X_test, y_test)
    print(f"Model accuracy: {forest_accuracy:.2f}")

    # neural network binary classifier
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(8, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(1, activation='sigmoid')
    ])

    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    nn_prediction = nn_model.predict(X_test)
    print("Neural network predictions made successfully.")

    nn_accuracy = nn_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"Neural network accuracy: {nn_accuracy:.2f}")