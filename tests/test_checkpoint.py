import json
import tempfile
import unittest
from pathlib import Path

from nn.linear import Linear
from nn.neuron import Neuron
from training.checkpoint import (
    CheckpointError,
    load_checkpoint,
    load_linear,
    load_neuron,
    save_checkpoint,
    save_linear,
    save_neuron,
)
from training.loop import train_epoch


class CheckpointSchemaTests(unittest.TestCase):
    def test_missing_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(CheckpointError):
                load_neuron(Path(directory) / "missing.json")

    def test_invalid_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text("{not-json", encoding="utf-8")

            with self.assertRaises(CheckpointError):
                load_neuron(path)

    def test_invalid_format_and_version_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for payload in (
                {"format": "wrong", "version": 1},
                {"format": "kinera.checkpoint", "version": 999},
            ):
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(CheckpointError):
                    load_neuron(path)

    def test_invalid_model_shape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(
                json.dumps(
                    {
                        "format": "kinera.checkpoint",
                        "version": 1,
                        "model": {
                            "type": "neuron",
                            "input_size": 2,
                            "activation": "relu",
                            "negative_slope": 0.01,
                            "weights": [0.1],
                            "bias": 0.2,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(CheckpointError):
                load_neuron(path)


class NeuronCheckpointTests(unittest.TestCase):
    def test_neuron_round_trip_preserves_parameters_activation_and_metadata(self) -> None:
        neuron = Neuron(
            input_size=2,
            seed=7,
            activation="leaky_relu",
            negative_slope=0.05,
        )
        neuron.weights = [0.4, -0.2]
        neuron.bias = 0.3
        neuron.forward([1.5, -0.5])
        original_output = neuron.output
        training_state = {
            "epoch": 12,
            "steps": 48,
            "best_validation_loss": 0.125,
        }
        metadata = {"experiment": "unit-test", "seed": 7}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neuron.json"
            saved_path = save_neuron(
                path,
                neuron,
                training_state=training_state,
                metadata=metadata,
            )
            loaded, info = load_neuron(saved_path)

        self.assertEqual(loaded.weights, neuron.weights)
        self.assertEqual(loaded.bias, neuron.bias)
        self.assertEqual(loaded.activation, "leaky_relu")
        self.assertEqual(loaded.negative_slope, 0.05)
        self.assertEqual(loaded.forward([1.5, -0.5]), original_output)
        self.assertEqual(info["training"], training_state)
        self.assertEqual(info["metadata"], metadata)

    def test_loaded_neuron_has_no_pending_transient_state(self) -> None:
        neuron = Neuron(input_size=1, seed=7)
        neuron.forward([1.0])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neuron.json"
            save_neuron(path, neuron)
            loaded, _ = load_neuron(path)

        self.assertIsNone(loaded.last_input)
        self.assertIsNone(loaded.last_z)
        self.assertIsNone(loaded.output)
        self.assertEqual(loaded.grad_weights, [0.0])
        self.assertEqual(loaded.grad_bias, 0.0)
        with self.assertRaises(RuntimeError):
            loaded.step(0.1)


class LinearCheckpointTests(unittest.TestCase):
    def test_linear_round_trip_preserves_parameters_and_output(self) -> None:
        layer = Linear(input_size=2, output_size=2, seed=7)
        layer.weights = [[0.5, -1.0], [2.0, 0.25]]
        layer.bias = [0.25, -0.5]
        original_output = layer.forward([2.0, -1.0])
        training_state = {"epoch": 4, "steps": 16}
        metadata = {"model": "linear-test"}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.json"
            save_linear(
                path,
                layer,
                training_state=training_state,
                metadata=metadata,
            )
            loaded, info = load_linear(path)

        self.assertEqual(loaded.weights, layer.weights)
        self.assertEqual(loaded.bias, layer.bias)
        self.assertEqual(loaded.forward([2.0, -1.0]), original_output)
        self.assertEqual(info["training"], training_state)
        self.assertEqual(info["metadata"], metadata)

    def test_generic_save_and_load_dispatch_by_model_type(self) -> None:
        neuron = Neuron(input_size=1, seed=7, activation="linear")
        layer = Linear(input_size=1, output_size=1, seed=7)

        with tempfile.TemporaryDirectory() as directory:
            neuron_path = Path(directory) / "neuron.json"
            linear_path = Path(directory) / "linear.json"
            save_checkpoint(neuron_path, neuron)
            save_checkpoint(linear_path, layer)
            loaded_neuron, _ = load_checkpoint(neuron_path)
            loaded_layer, _ = load_checkpoint(linear_path)

        self.assertIsInstance(loaded_neuron, Neuron)
        self.assertIsInstance(loaded_layer, Linear)

    def test_loader_rejects_wrong_model_type(self) -> None:
        neuron = Neuron(input_size=1, seed=7)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neuron.json"
            save_neuron(path, neuron)
            with self.assertRaises(CheckpointError):
                load_linear(path)


class CheckpointContinuationTests(unittest.TestCase):
    def test_save_creates_parent_and_replaces_existing_checkpoint(self) -> None:
        first = Neuron(input_size=1, seed=7, activation="linear")
        second = Neuron(input_size=1, seed=8, activation="linear")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "weights.json"
            save_neuron(path, first)
            save_neuron(path, second)
            loaded, _ = load_neuron(path)
            temporary_files = list(path.parent.glob(f".{path.name}.*.tmp"))

        self.assertEqual(loaded.weights, second.weights)
        self.assertEqual(loaded.bias, second.bias)
        self.assertEqual(temporary_files, [])

    def test_loaded_neuron_can_continue_training(self) -> None:
        samples = [([1.0], 2.0), ([2.0], 4.0)]
        original = Neuron(input_size=1, seed=7, activation="leaky_relu")
        original.weights = [0.5]
        original.bias = 0.5
        for _ in range(10):
            train_epoch(original, samples, 0.05, shuffle=False)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "neuron.json"
            save_neuron(path, original, training_state={"epoch": 10, "steps": 20})
            restored, info = load_neuron(path)

        self.assertEqual(info["training"]["epoch"], 10)
        for _ in range(10):
            train_epoch(original, samples, 0.05, shuffle=False)
            train_epoch(restored, samples, 0.05, shuffle=False)

        self.assertEqual(restored.weights, original.weights)
        self.assertEqual(restored.bias, original.bias)

    def test_loaded_linear_can_continue_training(self) -> None:
        samples = [
            ([1.0, 1.0], [2.0, 3.0]),
            ([2.0, 1.0], [3.0, 4.0]),
        ]
        original = Linear(input_size=2, output_size=2, seed=7)
        original.weights = [[0.5, 0.5], [0.5, 0.5]]
        original.bias = [0.5, 0.5]
        for _ in range(10):
            for inputs, targets in samples:
                prediction = original.forward(inputs)
                original.backward([
                    actual - target
                    for actual, target in zip(prediction, targets)
                ])
                original.step(0.05)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "linear.json"
            save_linear(path, original, training_state={"epoch": 10})
            restored, _ = load_linear(path)

        for _ in range(10):
            for model in (original, restored):
                for inputs, targets in samples:
                    prediction = model.forward(inputs)
                    model.backward([
                        actual - target
                        for actual, target in zip(prediction, targets)
                    ])
                    model.step(0.05)

        self.assertEqual(restored.weights, original.weights)
        self.assertEqual(restored.bias, original.bias)


if __name__ == "__main__":
    unittest.main()
