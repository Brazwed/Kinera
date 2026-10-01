import random
import unittest

from train_5min import build_samples, target_function


class TimedTrainingAdapterTests(unittest.TestCase):
    def test_build_samples_uses_the_declared_target_function(self) -> None:
        samples = build_samples(random.Random(7), count=4)

        self.assertEqual(len(samples), 4)
        for inputs, target in samples:
            self.assertEqual(target, target_function(inputs[0], inputs[1]))


if __name__ == "__main__":
    unittest.main()
