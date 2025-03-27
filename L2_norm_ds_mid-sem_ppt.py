import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data (X: years of experience, y: salary)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([2, 2.8, 3.6, 4.5, 5.1, 7.5, 9.2, 6.8, 11, 13])

# Fit Linear Regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Compute MSE
mse = mean_squared_error(y, y_pred)

# Compute L2 norm of residuals (error vector)
residuals = y - y_pred
l2_norm = np.linalg.norm(residuals, 2)

# Print results
print(f"MSE: {mse}")
print(f"L2 Norm of Residuals: {l2_norm}")

# Plot data and regression line
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Best Fit Line')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()
