# -*- coding: utf-8 -*-
"""Untitled18.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1B1yyy-d4Kzp6CitALtU-62cfJ7zZatsv
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = SGDRegressor(max_iter=1, warm_start=True, penalty=None, learning_rate='constant', eta0=0.01)

# Arrays to store training and validation errors
train_errors, val_errors = [], []

# Train the model
for epoch in range(1, 301):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_error)
    val_errors.append(val_error)

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_errors, label='Training Error')
plt.plot(val_errors, label='Validation Error')
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Analyze learning curves
train_errors_diff = np.diff(train_errors)
val_errors_diff = np.diff(val_errors)

if val_errors[-1] < val_errors[0]:
    print("The model is improving. It may be approaching optimal performance.")
else:
    print("The model performance is not improving. Consider early stopping.")

if np.all(val_errors_diff > 0) and np.all(train_errors_diff > 0):
    print("Both training and validation errors are increasing: Potential overfitting.")
elif np.all(val_errors_diff < 0) and np.all(train_errors_diff < 0):
    print("Both training and validation errors are decreasing: Model is learning well.")
elif train_errors[-1] < val_errors[-1]:
    print("Training error is much lower than validation error: Potential overfitting.")
else:
    print("Errors are relatively stable: Model may be at an optimal point or experiencing some instability.")

# Recommendations
if val_errors[-1] < val_errors[0]:
    print("Continue training, but monitor validation error closely for signs of overfitting.")
else:
    print("Consider stopping training and revising the model or hyperparameters.")