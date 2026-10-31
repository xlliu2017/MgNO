import argparse
import unittest

from experiment_args import smoothing_pair


class SmoothingPairTests(unittest.TestCase):
    def test_legacy_and_comma_separated_pairs(self):
        self.assertEqual(smoothing_pair("10"), [1, 0])
        self.assertEqual(smoothing_pair("11"), [1, 1])
        self.assertEqual(smoothing_pair("12,3"), [12, 3])

    def test_invalid_pairs(self):
        for value in ("1", "123", "-1,0", "1,", "1,2,3", "a0"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                smoothing_pair(value)
