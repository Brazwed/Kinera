"""Stateless CPU backend preserving the original Python arithmetic."""

from __future__ import annotations

from kinera.backends.cpu.ops import activations, embedding, linear, probability


class CPUBackend:
    """Python-standard-library oracle for the model's numerical operations."""

    name = "cpu"

    def linear_forward(self, weights, bias, inputs):
        return linear.forward(weights, bias, inputs)

    def linear_backward(self, weights, inputs, grad_output):
        return linear.backward(weights, inputs, grad_output)

    def linear_step(self, weights, bias, grad_weights, grad_bias, learning_rate):
        return linear.step(weights, bias, grad_weights, grad_bias, learning_rate)

    def embedding_forward(self, weights, ids):
        return embedding.forward(weights, ids)

    def embedding_backward(self, grad_weights, ids, grad_output):
        return embedding.backward(grad_weights, ids, grad_output)

    def embedding_step(self, weights, grad_weights, learning_rate):
        return embedding.step(weights, grad_weights, learning_rate)

    def neuron_forward(self, weights, bias, inputs, activation, negative_slope):
        z = sum(input_value * weight for input_value, weight in zip(inputs, weights)) + bias
        return z, activations.forward(activation, z, negative_slope)

    def neuron_backward(
        self, weights, inputs, z, grad_output, activation, negative_slope
    ):
        grad_z = float(grad_output) * activations.derivative(
            activation, z, negative_slope
        )
        grad_weights = [grad_z * input_value for input_value in inputs]
        grad_bias = grad_z
        grad_inputs = [grad_z * weight for weight in weights]
        return grad_weights, grad_bias, grad_inputs

    def neuron_step(self, weights, bias, grad_weights, grad_bias, learning_rate):
        weights = [
            weight - float(learning_rate) * gradient
            for weight, gradient in zip(weights, grad_weights)
        ]
        bias -= float(learning_rate) * grad_bias
        return weights, bias

    def activation(self, name, value, negative_slope=0.01):
        return activations.forward(name, value, negative_slope)

    def activation_derivative(self, name, value, negative_slope=0.01):
        return activations.derivative(name, value, negative_slope)

    def softmax(self, logits):
        return probability.softmax(logits)

    def cross_entropy(self, logits, target_id):
        return probability.cross_entropy(logits, target_id)

    def cross_entropy_gradient(self, logits, target_id):
        return probability.cross_entropy_gradient(logits, target_id)

    def binary_cross_entropy_with_logits(self, logit, target):
        return probability.binary_cross_entropy_with_logits(logit, target)

    def binary_cross_entropy_with_logits_gradient(self, logit, target):
        return probability.binary_cross_entropy_with_logits_gradient(logit, target)
