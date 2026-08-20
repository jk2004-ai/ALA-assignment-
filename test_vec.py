from vec import Vec
import math


v1 = Vec([1, 2, 3])
v2 = Vec([4, 5, 6])

assert v1 + v2 == Vec([5, 7, 9])

assert v2 - v1 == Vec([3, 3, 3])

assert 2 * v1 == Vec([2, 4, 6])

assert -v1 == Vec([-1, -2, -3])

v3 = Vec([1, 2, 3])
v3 += Vec([4, 5, 6])
assert v3 == Vec([5, 7, 9])

v3 *= 2
assert v3 == Vec([10, 14, 18])

assert Vec.zeros(3) == Vec([0, 0, 0])

assert Vec.ones(3) == Vec([1, 1, 1])

v = Vec.uniform(100)
assert len(v) == 100
assert all(0 <= x <= 1 for x in v)

v4 = Vec([3, 4])
assert math.isclose(v4.norm(), 5.0)

print("All tests passed!")
