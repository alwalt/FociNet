import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam

class PatternLayer(Layer):
    """
    Custom Pattern Layer for the GRNN.
    Computes the Gaussian function based on the input, training patterns, and sigma.
    """
    def __init__(self, train_data, sigma, **kwargs):
        super(PatternLayer, self).__init__(**kwargs)
        self.train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
        self.sigma = sigma

    def call(self, inputs):
        # Compute squared Euclidean distances (D^2) between inputs and training patterns
        diff = tf.expand_dims(inputs, axis=1) - self.train_data
        distances = tf.reduce_sum(tf.square(diff), axis=-1)
        # Apply Gaussian function
        return tf.exp(-distances / (2 * self.sigma ** 2))


class SummationLayer(Layer):
    """
    Custom Summation Layer for the GRNN.
    Computes the numerator and denominator of the GRNN formula.
    """
    def __init__(self, train_targets, **kwargs):
        super(SummationLayer, self).__init__(**kwargs)
        self.train_targets = tf.convert_to_tensor(train_targets, dtype=tf.float32)

    def call(self, inputs):
        # Numerator: Sum of weighted target values
        numerator = tf.reduce_sum(inputs * self.train_targets, axis=1, keepdims=True)
        # Denominator: Sum of weights
        denominator = tf.reduce_sum(inputs, axis=1, keepdims=True)
        return tf.concat([numerator, denominator], axis=1)


def build_grnn(train_data, train_targets, input_dim, sigma):
    """
    Build the GRNN model using custom layers.

    Parameters:
    - train_data: Training feature data
    - train_targets: Training target values
    - input_dim: Number of input features
    - sigma: Spread parameter for the Gaussian function

    Returns:
    - GRNN model
    """
    # Input layer
    inputs = layers.Input(shape=(input_dim,))
    
    # Pattern layer
    pattern_layer = PatternLayer(train_data, sigma)(inputs)
    
    # Summation layer
    summation_layer = SummationLayer(train_targets)(pattern_layer)
    
    # Output layer (Compute numerator/denominator)
    outputs = layers.Lambda(lambda x: x[:, 0] / (x[:, 1] + 1e-8))(summation_layer)
    
    # Build the model
    model = models.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


# Example usage
if __name__ == "__main__":
    # Training data
    X_train = np.array([[2], [4], [6], [8]], dtype=np.float32)  # Training inputs
    Y_train = np.array([3, 5, 7, 9], dtype=np.float32)  # Training outputs

    # Test input
    X_test = np.array([[5]], dtype=np.float32)  # Single input to predict

    # Hyperparameters
    input_dim = 1  # Single feature
    sigma = 1.0  # Spread parameter

    # Build the GRNN model
    grnn_model = build_grnn(X_train, Y_train, input_dim, sigma)

    # Display the model summary
    grnn_model.summary()

    # Train the model (epochs=1, as GRNN is non-iterative)
    grnn_model.fit(X_train, Y_train, epochs=1, batch_size=len(X_train), verbose=1)

    # Predict the output for the test input
    predictions = grnn_model.predict(X_test)
    print(f"Predicted output for input {X_test.flatten()}: {predictions.flatten()}")
