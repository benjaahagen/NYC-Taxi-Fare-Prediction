# NYC-Taxi-Fare-Prediction
This repository contains the code and results of an analysis on New York City taxi fares. The primary objective of this project is to develop a predictive model to estimate taxi fares based on historical trip data. The secondary objective is to perform an exploratory data analysis (EDA) to understand the patterns in taxi demand and the factors influencing the fare amounts.

## Dataset
The dataset used in this project is obtained from the New York City Taxi and Limousine Commission (TLC). It includes details such as pickup and dropoff locations, trip distances, fare amounts, total amounts, and other related information.

## Analysis
The analysis includes:
Calculation of average daily demand for taxis
Identification of the day with the highest and lowest demand
Analysis of taxi demand in different time periods
Calculation of average revenue generated on weekdays and weekends

## Predictive Model
A linear regression model is developed to predict taxi fares. The model uses the 'fare_amount' feature as the primary predictor for the 'total_amount' variable. The model achieved an RMSE of 0.010586720859309798 and an R2 score of 0.9999993879527193, indicating a good fit.

## Visualization
The project includes various visualizations such as bar plots, heatmaps, scatter plots, and histograms to better understand the data and the results of the analysis.
