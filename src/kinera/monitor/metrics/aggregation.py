"""Population statistics preserving Kinera's established arithmetic order."""

import math


def mean_std(values: list[float]) -> dict[str, float]:
    """Return population mean and standard deviation using the legacy formula."""
    if not values:
        raise ValueError("values não pode ser vazio")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance)}
