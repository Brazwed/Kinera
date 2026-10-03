"""CPU embedding lookup and update operations."""


def forward(weights: list[list[float]], ids: list[int]) -> list[list[float]]:
    return [weights[index].copy() for index in ids]


def backward(
    grad_weights: list[list[float]], ids: list[int], grad_output: list[list[float]]
) -> list[list[float]]:
    accumulated = [row.copy() for row in grad_weights]
    for position, index in enumerate(ids):
        accumulated[index] = [
            current + float(gradient)
            for current, gradient in zip(accumulated[index], grad_output[position])
        ]
    return accumulated


def step(
    weights: list[list[float]],
    grad_weights: list[list[float]],
    learning_rate: float,
) -> list[list[float]]:
    return [
        [
            weight - float(learning_rate) * gradient
            for weight, gradient in zip(row, gradient_row)
        ]
        for row, gradient_row in zip(weights, grad_weights)
    ]
