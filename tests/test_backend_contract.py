"""Tests for the backend contract and the CPU reference implementation."""

import importlib
from importlib.util import find_spec
import math
import unittest

from kinera.backends.cpu.backend import CPUBackend
from kinera.nn.losses.binary_cross_entropy import (
    binary_cross_entropy_with_logits,
    binary_cross_entropy_with_logits_gradient,
)
from kinera.nn.losses.cross_entropy import (
    _cross_entropy_gradient_validated,
    _cross_entropy_validated,
    _softmax_validated,
)
from kinera.nn.probability import cross_entropy, cross_entropy_gradient, softmax


class CPUBackendTests(unittest.TestCase):
    def _backend(self):
        try:
            spec = find_spec("kinera.backends.cpu.backend")
        except ModuleNotFoundError:
            spec = None
        self.assertIsNotNone(spec, "CPUBackend must be provided by kinera.backends")
        module = importlib.import_module("kinera.backends.cpu.backend")
        return module.CPUBackend()

    def test_cpu_backend_implements_contract_and_linear_math(self) -> None:
        backend = self._backend()
        contract = importlib.import_module("kinera.backends.base").Backend

        self.assertIsInstance(backend, contract)
        self.assertEqual(
            backend.linear_forward(
                [[2.0, -1.0], [0.5, 3.0]],
                [0.25, -0.5],
                [4.0, 2.0],
            ),
            [6.25, 7.5],
        )
        self.assertEqual(
            backend.linear_backward(
                [[2.0, -1.0], [0.5, 3.0]],
                [4.0, 2.0],
                [1.5, -2.0],
            ),
            ([[6.0, 3.0], [-8.0, -4.0]], [1.5, -2.0], [2.0, -7.5]),
        )
        self.assertEqual(
            backend.linear_step(
                [[2.0, -1.0], [0.5, 3.0]],
                [0.25, -0.5],
                [[6.0, 3.0], [-8.0, -4.0]],
                [1.5, -2.0],
                0.1,
            ),
            ([[1.4, -1.3], [1.3, 3.4]], [0.25 - 0.1 * 1.5, -0.5 - 0.1 * -2.0]),
        )

    def test_cpu_backend_preserves_embedding_copy_accumulation_and_update(self) -> None:
        backend = self._backend()
        weights = [[1.0, 2.0], [3.0, 4.0]]
        output = backend.embedding_forward(weights, [1, 0, 1])
        self.assertEqual(output, [[3.0, 4.0], [1.0, 2.0], [3.0, 4.0]])
        output[0][0] = -99.0
        self.assertEqual(weights, [[1.0, 2.0], [3.0, 4.0]])

        gradients = backend.embedding_backward(
            [[0.0, 0.0], [0.0, 0.0]],
            [1, 0, 1],
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        )
        self.assertEqual(gradients, [[0.3, 0.4], [0.6, 0.8]])
        self.assertEqual(
            backend.embedding_step(weights, gradients, 0.1),
            [[0.97, 1.96], [2.94, 3.92]],
        )

    def test_cpu_backend_probability_and_activation_operations(self) -> None:
        backend = self._backend()
        probabilities = backend.softmax([1.0, 2.0])
        self.assertAlmostEqual(probabilities[0], 1.0 / (1.0 + math.e))
        self.assertAlmostEqual(probabilities[1], math.e / (1.0 + math.e))
        self.assertAlmostEqual(
            backend.cross_entropy([1.0, 2.0], 1),
            math.log1p(math.exp(-1.0)),
        )
        self.assertEqual(
            backend.cross_entropy_gradient([1.0, 2.0], 1),
            [probabilities[0], probabilities[1] - 1.0],
        )
        self.assertEqual(backend.activation("relu", -2.0), 0.0)
        self.assertEqual(backend.activation_derivative("linear", -2.0), 1.0)


class ProbabilityBackendInjectionTests(unittest.TestCase):
    def test_public_probability_apis_use_the_selected_backend(self) -> None:
        class RecordingBackend(CPUBackend):
            def __init__(self) -> None:
                self.calls: list[str] = []

            def softmax(self, logits):
                self.calls.append("softmax")
                return super().softmax(logits)

            def cross_entropy(self, logits, target_id):
                self.calls.append("cross_entropy")
                return super().cross_entropy(logits, target_id)

            def cross_entropy_gradient(self, logits, target_id):
                self.calls.append("cross_entropy_gradient")
                return super().cross_entropy_gradient(logits, target_id)

        backend = RecordingBackend()
        softmax([0.0, 1.0], backend=backend)
        cross_entropy([0.0, 1.0], 1, backend=backend)
        cross_entropy_gradient([0.0, 1.0], 1, backend=backend)

        self.assertEqual(
            backend.calls,
            ["softmax", "cross_entropy", "cross_entropy_gradient"],
        )


class ExtendedBackendInjectionTests(unittest.TestCase):
    class RecordingBackend(CPUBackend):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def softmax(self, logits):
            self.calls.append("softmax")
            return super().softmax(logits)

        def cross_entropy(self, logits, target_id):
            self.calls.append("cross_entropy")
            return super().cross_entropy(logits, target_id)

        def cross_entropy_gradient(self, logits, target_id):
            self.calls.append("cross_entropy_gradient")
            return super().cross_entropy_gradient(logits, target_id)

        def binary_cross_entropy_with_logits(self, logit, target):
            self.calls.append("binary_cross_entropy_with_logits")
            return super().binary_cross_entropy_with_logits(logit, target)

        def binary_cross_entropy_with_logits_gradient(self, logit, target):
            self.calls.append("binary_cross_entropy_with_logits_gradient")
            return super().binary_cross_entropy_with_logits_gradient(logit, target)

    def test_standalone_binary_cross_entropy_uses_injected_backend(self) -> None:
        backend = self.RecordingBackend()
        binary_cross_entropy_with_logits(2.0, 1.0, backend=backend)
        binary_cross_entropy_with_logits_gradient(2.0, 1.0, backend=backend)
        self.assertEqual(
            backend.calls,
            ["binary_cross_entropy_with_logits", "binary_cross_entropy_with_logits_gradient"],
        )

    def test_validated_probability_helpers_use_injected_backend(self) -> None:
        backend = self.RecordingBackend()
        logits = [0.0, 1.0]
        _softmax_validated(logits, backend=backend)
        _cross_entropy_validated(logits, 1, backend=backend)
        _cross_entropy_gradient_validated(logits, 1, backend=backend)
        self.assertEqual(
            backend.calls,
            ["softmax", "cross_entropy", "cross_entropy_gradient"],
        )


if __name__ == "__main__":
    unittest.main()
