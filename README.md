# FPL Machine Learning Project

This project uses historical Fantasy Premier League (FPL) data to build predictive models that help with player selection decisions.

## Problem 1: Predicting Fantasy Points for a Gameweek [Linear Regression]

Given player stats for a particular gameweek (e.g., GW38), predict each player's `total_points` using other stats from the same gameweek. This is a regression problem where the model is trained and evaluated on different players from the same week.

## Problem 2: Predicting Player Position (softmax Multi-class Classification)

Predict a player's position (**Goalkeeper**, **Defender**, **Midfielder**, or **Forward**) based on gameweek-specific stats. Formulate this as a multi-class classification problem using Softmax Regression.

