import tensorflow as tf
import numpy as np

class NeuralNet():
    def __init__(self, input_shape, output_shape):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.input_layer = tf.keras.layers.InputLayer(input_shape)
        self.output_layer = tf.keras.layers.Dense(output_shape, activation='sigmoid')
        self.opt = tf.keras.optimizers.SGD(learning_rate=0.1)

    def forward_pass(self, inputs, target):
        with tf.GradientTape() as tape:
            # Call model on a test input
            tape.watch(self.output_layer.weights)
            result = (target - self.output_layer(self.input_layer(inputs))) ** 2
            grads = tape.gradient(result, self.output_layer.weights)
        return (result, grads)

    def backpropagate(self, input_data, label):
        result, gradient = self.forward_pass(input_data,label)
        # When we get the gradient, the backprop value is the sum of all the grads.
        # We must div by number of samples
        self.opt.apply_gradients(zip(gradient,self.output_layer.weights))
        return np.array(gradient[0]) * self.output_layer.weights[0] / input_data.T, result
        # return gradient[0],self.output_layer.weights[0],input_data