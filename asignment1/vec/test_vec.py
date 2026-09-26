import unittest
from vec import Vec


class TestVec(unittest.TestCase):
##mean tested here 
    def test_mean_positive(self):
        v = Vec([10, 20, 30])
        self.assertEqual(v.mean(), 20)

    def test_mean_negative(self):
        v = Vec([-10, -20, -30])
        self.assertEqual(v.mean(), -20)

    def test_mean_mixed(self):
        v = Vec([-5, 0, 5])
        self.assertEqual(v.mean(), 0)

    def test_mean_decimal(self):
        v = Vec([1.5, 2.5, 3.5])
        self.assertAlmostEqual(v.mean(), 2.5)


 #demean tested here 

    def test_demean_positive(self):
        v = Vec([2, 4, 6])
        result = v.demean()

        self.assertEqual(result.elements, [-2, 0, 2])

    def test_demean_negative(self):
        v = Vec([-2, -4, -6])
        result = v.demean()

        self.assertEqual(result.elements, [2, 0, -2])

    def test_demean_decimal(self):
        v = Vec([1.5, 2.5, 3.5])
        result = v.demean()

        self.assertEqual(result.elements, [-1.0, 0.0, 1.0])

    def test_demean_mean_is_zero(self):
        v = Vec([10, 20, 30, 40])
        result = v.demean()

        self.assertAlmostEqual(result.mean(), 0)


#std tested here 

    def test_std_simple(self):
        v = Vec([1, 2, 3])
        self.assertAlmostEqual(v.std(), 0.8164965809)

    def test_std_constant_values(self):
        v = Vec([5, 5, 5, 5])
        self.assertEqual(v.std(), 0)

    def test_std_negative_values(self):
        v = Vec([-2, -1, 0, 1, 2])
        self.assertAlmostEqual(v.std(), 1.414213562)



    def test_demean_does_not_change_original(self):
        v = Vec([1, 2, 3])

        v.demean()

        self.assertEqual(v.elements, [1, 2, 3])

#when we gave same elemet 
    def test_single_element(self):
        v = Vec([100])

        self.assertEqual(v.mean(), 100)
        self.assertEqual(v.demean().elements, [0])
        self.assertEqual(v.std(), 0)


if __name__ == "__main__":
    result = unittest.main()

    if result:
        print("\nALL TEST CASES PASSED!")
    else:
        print("\nSOME TEST CASES FAILED!")