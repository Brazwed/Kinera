"""Validated-wrapper arithmetic for CPU probability and loss operations."""

import math


def softmax(logits: list[float]) -> list[float]:
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def cross_entropy(logits: list[float], target_id: int) -> float:
    maximum = max(logits)
    log_sum_exp = maximum + math.log(
        sum(math.exp(value - maximum) for value in logits)
    )
    return log_sum_exp - logits[target_id]


def cross_entropy_gradient(logits: list[float], target_id: int) -> list[float]:
    probabilities = softmax(logits)
    probabilities[target_id] -= 1.0
    return probabilities


def binary_cross_entropy_with_logits(logit: float, target: float) -> float:
    return max(logit, 0.0) - logit * target + math.log1p(math.exp(-abs(logit)))


def binary_cross_entropy_with_logits_gradient(logit: float, target: float) -> float:
    if logit >= 0.0:
        exponential = math.exp(-logit)
        sigmoid = 1.0 / (1.0 + exponential)
    else:
        exponential = math.exp(logit)
        sigmoid = exponential / (1.0 + exponential)
    return sigmoid - target
