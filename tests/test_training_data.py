from collections import Counter
import copy
import unittest

from nn.neuron import Neuron
from training.data import generate_samples, split_samples, target_function
from training.loop import TrainingConfig, fit, train_epoch


class TrainingDataTests(unittest.TestCase):
    def test_target_function_returns_expected_values(self) -> None:
        self.assertEqual(target_function(1.0, 2.0), 9.0)
        self.assertEqual(target_function(2.0, 1.0), 8.0)

    def test_generation_is_deterministic_and_uses_target_function(self) -> None:
        first = generate_samples(count=8, seed=7)
        second = generate_samples(count=8, seed=7)
        different = generate_samples(count=8, seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(first), 8)
        for inputs, target in first:
            self.assertEqual(target, target_function(inputs[0], inputs[1]))
            self.assertGreaterEqual(inputs[0], 0.25)
            self.assertLessEqual(inputs[0], 4.0)
            self.assertGreaterEqual(inputs[1], 0.25)
            self.assertLessEqual(inputs[1], 4.0)

    def test_generation_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(ValueError):
            generate_samples(count=0, seed=7)
        with self.assertRaises(TypeError):
            generate_samples(count=True, seed=7)
        with self.assertRaises(ValueError):
            generate_samples(count=2, seed=7, low=4.0, high=4.0)
        with self.assertRaises(ValueError):
            generate_samples(count=2, seed=7, low=5.0, high=4.0)

    def test_split_is_reproducible_and_covers_all_samples(self) -> None:
        samples = generate_samples(count=10, seed=7)

        first_train, first_validation = split_samples(
            samples, validation_ratio=0.3, seed=42
        )
        second_train, second_validation = split_samples(
            samples, validation_ratio=0.3, seed=42
        )

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_train), 7)
        self.assertEqual(len(first_validation), 3)
        original_counter = Counter((tuple(inputs), target) for inputs, target in samples)
        split_counter = Counter(
            (tuple(inputs), target)
            for inputs, target in first_train + first_validation
        )
        self.assertEqual(split_counter, original_counter)

    def test_split_does_not_mutate_or_alias_original_samples(self) -> None:
        samples = generate_samples(count=6, seed=7)
        original = copy.deepcopy(samples)

        train_samples, validation_samples = split_samples(
            samples, validation_ratio=0.5, seed=42
        )
        train_samples[0][0][0] += 100.0

        self.assertEqual(samples, original)
        original_input_ids = {id(inputs) for inputs, _ in samples}
        split_input_ids = {
            id(inputs) for inputs, _ in train_samples + validation_samples
        }
        self.assertTrue(original_input_ids.isdisjoint(split_input_ids))

    def test_split_rejects_invalid_ratios_and_datasets(self) -> None:
        samples = generate_samples(count=4, seed=7)

        with self.assertRaises(ValueError):
            split_samples(samples, validation_ratio=0.0, seed=42)
        with self.assertRaises(ValueError):
            split_samples(samples, validation_ratio=1.0, seed=42)
        with self.assertRaises(ValueError):
            split_samples([], validation_ratio=0.5, seed=42)
        with self.assertRaises(ValueError):
            split_samples([samples[0]], validation_ratio=0.5, seed=42)

    def test_validation_samples_do_not_update_model_parameters(self) -> None:
        samples = generate_samples(count=10, seed=7)
        train_samples, validation_samples = split_samples(
            samples, validation_ratio=0.3, seed=42
        )
        config = TrainingConfig(
            learning_rate=0.01,
            max_epochs=1,
            patience=10,
            shuffle=False,
            seed=42,
        )
        fit_neuron = Neuron(input_size=2, seed=7)
        manual_neuron = Neuron(input_size=2, seed=7)
        fit_neuron.weights = manual_neuron.weights = [0.5, 0.5]
        fit_neuron.bias = manual_neuron.bias = 0.5

        fit(fit_neuron, train_samples, validation_samples, config)
        train_epoch(
            manual_neuron,
            train_samples,
            config.learning_rate,
            shuffle=False,
        )

        self.assertEqual(fit_neuron.weights, manual_neuron.weights)
        self.assertEqual(fit_neuron.bias, manual_neuron.bias)


if __name__ == "__main__":
    unittest.main()
