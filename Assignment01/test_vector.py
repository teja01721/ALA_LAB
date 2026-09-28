import unittest
import math

from vector import Vector


class TestVector(unittest.TestCase):

    def test_mean(self):
        v = Vector([1, 2, 3, 4, 5])

        self.assertAlmostEqual(v.mean(), 3.0)

    def test_demean(self):
        v = Vector([1, 2, 3, 4, 5])

        result = v.demean()

        expected = Vector([-2, -1, 0, 1, 2])

        self.assertEqual(result._entries, expected._entries)

    def test_demean_mean_is_zero(self):
        v = Vector([10, 20, 30, 40])

        demeaned = v.demean()

        self.assertAlmostEqual(demeaned.mean(), 0.0)

    def test_std(self):
        v = Vector([1, 2, 3, 4, 5])

        expected = math.sqrt(2)

        self.assertAlmostEqual(v.std(), expected)

    def test_std_nonnegative(self):
        v = Vector([2, 4, 6, 8])

        self.assertGreaterEqual(v.std(), 0.0)


if __name__ == "__main__":
    unittest.main()
