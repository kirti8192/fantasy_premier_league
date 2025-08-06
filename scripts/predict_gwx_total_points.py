import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def load_data():
    # import data
    df_raw_train = pd.read_csv('datasets/vaastav/2024_2025/fpl_data_all_gw.csv')
    return df_raw_train

def get_df_gw(df_raw_train, gw_num):

    # get df for specific game week
    df_gw = pick_gw_cols(df_raw_train, gw_num)

    # drop unnecessary columns
    df_gw.drop(columns=['element', 'name', 'team'], inplace=True)
    gw_cols_to_drop = ['fixture', 'opponent_team', 'value']
    df_gw.drop(columns=[f'gw{gw_num}_{col}' for col in gw_cols_to_drop], inplace=True)

    # one-hot encode position
    position_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    pos_onehot = position_encoder.fit_transform(df_gw[['position']])
    pos_onehot_df = pd.DataFrame(pos_onehot, columns=position_encoder.get_feature_names_out(['position']))

    # Align indices before concatenation
    pos_onehot_df.index = df_gw.index

    # Drop the original column and concatenate
    df_gw = pd.concat([df_gw.drop(columns=['position']), pos_onehot_df], axis=1)
    return df_gw

def pick_gw_cols(df, gw_num):

    # retain common columns
    common_cols = [col for col in df.columns if not col.startswith('gw')]

    # retain only columns for the specified game week
    gw_cols = [col for col in df.columns if col.startswith(f'gw{gw_num}_')]

    selected_cols = common_cols + gw_cols
    return df[selected_cols].copy()

def compute_cost(X, y, theta):
    theta = np.array(theta).flatten()  # in case it's a column vector
    return 0.5*sum((X@theta - y.values.ravel())**2)/ len(y)    # MSE

def plot_costs(X, y, history):
    # Compute cost at each step
    costs = [compute_cost(X, y, history[i]) for i in range(len(history))]

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(costs, label='Cost', color='red')
    plt.xlabel("Iteration")
    plt.ylabel("Cost (MSE)")
    plt.title("Cost vs Iteration")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def batch_gradient_descent_train(X, y, alpha, num_iterations):
    
    # insert bias term
    X.insert(0, 'bias', 1)

    # initialize
    theta = np.array([0]*(X.shape[1]))

    # batch gradient descent
    history_bgd = np.zeros((num_iterations + 1, X.shape[1]))
    history_bgd[0,:] = theta

    # Perform batch gradient descent
    for iter in range(num_iterations):
        prediction = X @ theta
        error = prediction - y.values.ravel()
        gradient = (X.T @ error / len(y)).to_numpy()
        theta = theta - alpha * gradient
        history_bgd[iter+1, :] = theta

    # plot_costs(X, y, history_bgd)

    return theta, history_bgd

def batch_gradient_descent_predict(X, theta):
    
    # insert bias term
    X.insert(0, 'bias', 1)

    return X @ theta

def analyze_predictions(y_true, y_pred, player_labels=None, do_plots=False):
    # Combine true and predicted values into a DataFrame
    results_df = pd.DataFrame({
        'True': y_true.values.ravel(),
        'Predicted': y_pred,
        'Residual': y_true.values.ravel() - y_pred
    })

    # Calculate metrics
    rmse = root_mean_squared_error(results_df['True'], results_df['Predicted'])
    mae = mean_absolute_error(results_df['True'], results_df['Predicted'])
    r2 = r2_score(results_df['True'], results_df['Predicted'])
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}")

    if do_plots:

        # 1. True vs Predicted
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='True', y='Predicted', data=results_df)
        plt.plot([results_df['True'].min(), results_df['True'].max()],
                [results_df['True'].min(), results_df['True'].max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted Values (R²={r2:.3f})')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 2. Residual Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(results_df['Residual'], bins=30, kde=True)
        plt.axvline(0, color='black', linestyle='--')
        plt.xlabel('Residuals')
        plt.title('Distribution of Residuals')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3. Residuals vs Predicted
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='Predicted', y='Residual', data=results_df)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residual')
        plt.title('Residuals vs Predicted')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 4. Error per Player (sorted)
        if player_labels is None:
            player_labels = results_df.index.astype(str)
        sorted_errors = results_df.assign(Player=player_labels).sort_values(by='Residual')
        plt.figure(figsize=(12, 6))
        colors = ['green' if e < 0 else 'red' for e in sorted_errors['Residual']]
        plt.bar(sorted_errors['Player'], sorted_errors['Residual'], color=colors)
        plt.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=90)
        plt.xlabel('Player')
        plt.ylabel('Residual')
        plt.title('Prediction Error per Player (Sorted)')
        plt.tight_layout()
        plt.show()

    return rmse, mae, r2

# params
gw_num = 1

# get data
df_raw_train = load_data()

# get gw-specific dataframe
df_gw = get_df_gw(df_raw_train, gw_num)

# split target from features
target_col = f'gw{gw_num}_total_points'
X = df_gw.drop(columns=[target_col])
y = df_gw[[target_col]]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=8)

# batch gradient descent
theta, history = batch_gradient_descent_train(X_train, y_train, alpha=0.0001, num_iterations=200)
predictions = batch_gradient_descent_predict(X_test, theta)
(rmse, mae, r2) = analyze_predictions(y_test, predictions, )