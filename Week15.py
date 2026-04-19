import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dataset 
X = np.array([[50], [60], [70], [80], [90]])
y = np.array([100, 120, 140, 160, 180])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
X_test = np.array([[75]])
prediction = model.predict(X_test)

print("Predicted price:", prediction[0])

# Evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
print("MSE:", mse)