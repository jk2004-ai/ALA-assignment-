import unittest
from vec import Vec


class TestVec(unittest.TestCase):

    def test_mean(self):
        v = Vec([1, 2, 3, 4, 5])

        self.assertEqual(v.mean(), 3)

    def test_demean(self):
        v = Vec([1, 2, 3, 4, 5])

        d = v.demean()

        expected = [-2, -1, 0, 1, 2]

        self.assertEqual(d.elements, expected)

    def test_demean_mean_zero(self):
        v = Vec([1, 2, 3, 4, 5])

        d = v.demean()

        self.assertAlmostEqual(d.mean(), 0)

    def test_demean_returns_new_vector(self):
        v = Vec([1, 2, 3])

        d = v.demean()

        self.assertIsNot(v, d)

        self.assertEqual(v.elements, [1, 2, 3])

    def test_std(self):
        v = Vec([1, 2, 3, 4, 5])

        self.assertAlmostEqual(
            v.std(),
            1.414213562
        )

    def test_constant_vector_std(self):
        v = Vec([5, 5, 5, 5])

        self.assertEqual(v.std(), 0)


if __name__ == "__main__":
    unittest.main()