import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Number of data points
num_samples = 1000

# Generate synthetic data
sqft = 2000 + 1000 * np.random.rand(num_samples, 1)  # Assuming a range of 2000 to 3000 sqft
num_rooms = np.random.randint(2, 7, size=(num_samples, 1))  # Assuming a range of 2 to 6 rooms
# Generate right-skewed ages using a gamma distribution
shape_param = 2  # Experiment with different shape parameters
age = np.random.gamma(shape_param, scale=10, size=(num_samples, 1)) + 1  # Adding 1 to avoid age being zero

# Combine input features into a matrix X
X = np.column_stack((sqft, num_rooms, age))

# Define the true coefficients for the linear regression model
true_coefficients = np.array([[150], [50], [30]])  # Adjust coefficients as needed

# Generate noise for the output variable
noise = 10000 * np.random.randn(num_samples, 1)  # Assuming a standard deviation of 10,000

# Generate the output variable (house prices) using the linear relationship with some noise
house_prices = X.dot(true_coefficients) + noise

# Create a DataFrame
data = pd.DataFrame(np.column_stack((sqft, num_rooms, age, house_prices)), columns=['sqft', 'num_rooms', 'age', 'house_prices'])

# Save the DataFrame to a CSV file
data.to_csv('house_prices_dataset.csv', index=False)

# Display the first few rows of the DataFrame
print(data.head())
