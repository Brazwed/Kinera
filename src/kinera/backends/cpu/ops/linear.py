"""CPU affine-layer arithmetic copied from the reference implementation."""

import operator


def forward(weights: list[list[float]], bias: list[float], inputs: list[float]) -> list[float]:
    return [
        sum(map(operator.mul, row, inputs)) + row_bias
        for row, row_bias in zip(weights, bias)
    ]


def backward(
    weights: list[list[float]], inputs: list[float], grad_output: list[float]
) -> tuple[list[list[float]], list[float], list[float]]:
    grad_output_float = [float(value) for value in grad_output]
    columns = [list(column) for column in zip(*weights)]
    grad_weights = [
        [grad_output_float[row_index] * input_value for input_value in inputs]
        for row_index in range(len(weights))
    ]
    grad_bias = grad_output_float.copy()
    grad_inputs = [
        float(sum(map(operator.mul, column, grad_output_float)))
        for column in columns
    ]
    return grad_weights, grad_bias, grad_inputs


def step(
    weights: list[list[float]],
    bias: list[float],
    grad_weights: list[list[float]],
    grad_bias: list[float],
    learning_rate: float,
) -> tuple[list[list[float]], list[float]]:
    rate = float(learning_rate)
    updated_weights = [
        [
            weight - rate * gradient
            for weight, gradient in zip(row, gradient_row)
        ]
        for row, gradient_row in zip(weights, grad_weights)
    ]
    updated_bias = [
        value - rate * gradient for value, gradient in zip(bias, grad_bias)
    ]
    return updated_weights, updated_bias
