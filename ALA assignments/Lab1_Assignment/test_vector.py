
import math
import pytest
from vec import Vec


def test_mean():
    v = Vec([10, -2, 8, 23, 6, 75])

    assert v.mean() == pytest.approx(20)


def test_mean_single_element():
    v = Vec([67])

    assert v.mean() == pytest.approx(67)


def test_mean_of_zero_vector():
    v = Vec([0, 0, 0])

    assert v.mean() == pytest.approx(0)


def test_mean_of_equal_values():
    v = Vec([67, 67, 67, 67, 67])

    assert v.mean() == pytest.approx(67)


def test_demean():
    v = Vec([10, -2, 8, 23, 6, 75])

    result = v.demean()

    expected = (-10.0, -22.0, -12.0, 3.0, -14.0, 55.0)

    assert result.elements == expected


def test_demean_mean_is_zero():
    v = Vec([10, -2, 8, 23, 6, 75])

    demeaned = v.demean()

    assert demeaned.mean() == pytest.approx(0)


def test_demean_sum_is_zero():
    v = Vec([10, -2, 8, 23, 6, 75])

    demeaned = v.demean()

    assert sum(demeaned.elements) == pytest.approx(0)


def test_demean_constant_vector():
    v = Vec([67, 67, 67, 67, 67])

    demeaned = v.demean()

    assert demeaned.elements == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_demean_does_not_modify_original():
    v = Vec([10, -2, 8, 23, 6, 75])

    original = v.elements

    v.demean()

    assert v.elements == original


def test_std():
    v = Vec([10, -2, 8, 23, 6, 75])

    assert v.std() == pytest.approx(25.68398, abs=0.00001)


def test_std_constant_vector():
    v = Vec([67, 67, 67, 67, 67])

    assert v.std() == pytest.approx(0)


def test_std_zero_vector():
    v = Vec([0, 0, 0, 0])

    assert v.std() == pytest.approx(0)


def test_std_is_non_negative():
    v = Vec([-67, -51, -95, -21])

    assert v.std() >= 0


def test_std_using_demeaned_vector():
    v = Vec([1, 2, 3, 4, 5])

    demeaned = v.demean()

    expected = math.sqrt(
        sum(x ** 2 for x in demeaned.elements) / len(v)
    )

    assert v.std() == pytest.approx(expected)

