"""CPU neuron arithmetic preserving the original Python operation order."""

from . import activations


def forward(weights, bias, inputs, activation, negative_slope):
    z = sum(input_value * weight for input_value, weight in zip(inputs, weights)) + bias
    return z, activations.forward(activation, z, negative_slope)


def backward(weights, inputs, z, grad_output, activation, negative_slope):
    grad_z = float(grad_output) * activations.derivative(
        activation, z, negative_slope
    )
    grad_weights = [grad_z * input_value for input_value in inputs]
    grad_bias = grad_z
    grad_inputs = [grad_z * weight for weight in weights]
    return grad_weights, grad_bias, grad_inputs


def step(weights, bias, grad_weights, grad_bias, learning_rate):
    weights = [
        weight - float(learning_rate) * gradient
        for weight, gradient in zip(weights, grad_weights)
    ]
    bias -= float(learning_rate) * grad_bias
    return weights, bias
