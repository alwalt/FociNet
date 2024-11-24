import numpy as np
import tensorflow as tf

# Training data
train_inputs = np.array([2, 4, 6, 8], dtype=np.float32)  # Input samples
train_outputs = np.array([3, 5, 7, 9], dtype=np.float32)  # Corresponding outputs

# Test input
test_input = 5.0

# Sigma value (spread parameter)
sigma = 1.0


def grnn_predict(train_inputs, train_outputs, test_input, sigma):
    """
    Implements GRNN prediction for a single test input.

    Parameters:
    - train_inputs: Array of training input samples.
    - train_outputs: Array of training output samples.
    - test_input: Single input value to predict the output for.
    - sigma: Spread parameter for the Gaussian function.

    Returns:
    - Predicted output for the test input.
    """
    # Step 1: Calculate squared Euclidean distances
    distances = np.square(train_inputs - test_input)

    # Step 2: Calculate weights using the Gaussian activation function
    weights = np.exp(-distances / (2 * sigma ** 2))

    # Step 3: Compute the numerator and denominator
    numerator = np.sum(weights * train_outputs)  # Weighted sum of outputs
    denominator = np.sum(weights)  # Sum of weights

    # Step 4: Compute the final prediction
    predicted_output = numerator / denominator
    return predicted_output


# Predict the output for the test input
predicted_output = grnn_predict(train_inputs, train_outputs, test_input, sigma)

print(f"Predicted output for input {test_input}: {predicted_output}")
