import timeit
from vec import Vec


sizes = [200000, 400000, 800000, 1600000, 3200000, 6400000]

print("Size     Addition       Subtraction     Multiplication     Norm")
print("-" * 70)

for n in sizes:

    v1 = Vec.uniform(n)
    v2 = Vec.uniform(n)

    # Number of repetitions
    repetitions = 100

    add_time = timeit.timeit(
        lambda: v1 + v2,
        number=repetitions
    ) / repetitions

    sub_time = timeit.timeit(
        lambda: v1 - v2,
        number=repetitions
    ) / repetitions

    mul_time = timeit.timeit(
        lambda: 2 * v1,
        number=repetitions
    ) / repetitions

    norm_time = timeit.timeit(
        lambda: v1.norm(),
        number=repetitions
    ) / repetitions

    print(
        f"{n:<8}"
        f"{add_time:<15.6f}"
        f"{sub_time:<16.6f}"
        f"{mul_time:<19.6f}"
        f"{norm_time:.6f}"
    )
