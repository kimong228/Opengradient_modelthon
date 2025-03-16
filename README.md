# ETHUSDT 1hour Predictor by XGBoost Model

# Overview

This model is an XGBoost regression model designed to predict the target variable target\_ETHUSDT using historical Ethereum data from the ETHUSDT\_1h\_spot\_forecast\_training.csv dataset. The model is optimized to minimize the root mean squared error (RMSE) and is converted to the ONNX format for efficient deployment and inference.

# Model Details

* **Algorithm**: XGBoost
* **Objective**: Regression
* **Parameters**:
  * objective: reg:squarederror (regression with squared error)
  * eval\_metric: RMSE (Root Mean Squared Error)
  * eta: 0.05 (learning rate)
  * max\_depth: 6 (maximum depth of a tree)
* **Training Data**: The dataset is split into 80% for training and 20% for validation to ensure the model's generalization capability.
* **Boosting Rounds**: 1000 with early stopping after 10 rounds if no improvement.

# Input

* **Features**: All columns except target\_ETHUSDT  are used as features. The feature columns are renamed to match XGBoost's expected pattern (f0, f1, ..., fN).
* **Input Shape**: The model expects input data in the shape of \[1, number\_of\_features] as a FloatTensorType.
* **Sample**:

```py
"candles":[[2678.89,2661.41,2670.9,2679.98,2669.34,2671.89,2662.16,2670.91,2686.28,2681.59,2679.27,2672.36,2683.7,2684.0,2674.19,2672.0,2673.3,2688.72,2695.27,2700.0,2651.26,2658.28,2658.99,2667.26,2658.0,2658.89,2637.71,2668.22,2675.6,2676.29,2661.41,2670.89,2679.99,2669.34,2671.88,2662.16,2670.91,2686.28,2681.59,2687.75,96865.79,96118.12,96319.22,96508.63,96228.97,96215.32,96026.01,96242.67,96376.78,96068.21,96884.89,96443.5,96533.52,96638.05,96280.0,96259.84,96328.14,96445.45,96463.2,96211.84,96046.18,96023.17,96060.5,96156.55,96032.99,96013.0,95800.0,96208.0,95900.94,95937.05,96118.12,96319.22,96508.63,96228.98,96215.33,96026.01,96242.66,96376.77,96068.22,96036.47,8.0]]
```

# Output

* **Prediction**: The model outputs a continuous value representing the predicted target variable target\_ETHUSDT.
* **Sample**:

```py
  "variable": [[0.00015973]]
```

# **Performance**

* **Validation Metric**: RMSE
* **Validation RMSE**: The RMSE is calculated and printed during the evaluation phase, providing a measure of the model's accuracy on the validation set.

# Limitations and Biases

* **Data Dependency**: The model's performance is highly dependent on the quality and representativeness of the training data. Any biases present in the historical data may be reflected in the model's predictions.
* **Market Conditions**: The model may not fully capture future market conditions, especially if they differ significantly from historical trends.

# Top Features

* The model utilizes all available features except target\_ETHUSDT.

# Link
https://hub.opengradient.ai/models/kmong/og-ethusdt-1h-return-xgb-model
