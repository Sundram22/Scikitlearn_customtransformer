import numpy as np
from sklearn.preprocessing import FunctionTransformer  # Fixed typo in import
from sklearn.datasets import make_regression  # Fixed typo in import
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Define the cubic transformation function
def cube_transform(x):
    return np.power(x, 3)

# Create the custom transformer
cube_transformer = FunctionTransformer(cube_transform)

# Generate some data
X, y = make_regression(n_samples=100, n_features=2, noise=0.1, random_state=42)

# Use the transformer directly
X_transformed = cube_transformer.fit_transform(X)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_transformed, y)

# Print the model coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
