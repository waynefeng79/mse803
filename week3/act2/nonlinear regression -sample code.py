import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate non-linear synthetic data
np.random.seed(42)
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# 2. Transform features to polynomial terms (Degree = 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# 3. Train the Linear Regression model on polynomial features
model = LinearRegression()
model.fit(X_poly, y)

# 4. Predict values for the original dataset
y_pred = model.predict(X_poly)

# 5. Make a prediction for a brand new data point
X_new = np.array([[1.5]])
X_new_poly = poly_features.transform(X_new)
new_prediction = model.predict(X_new_poly)
print(f"Prediction for X = 1.5: {new_prediction[0][0]:.4f}")

# 6. Evaluate the model performance
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")

# 7. Plot the regression curve
X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Fit (deg=2)')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
