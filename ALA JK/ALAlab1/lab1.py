def ho_add(x):
    return lambda y: x+y
f = ho_add(100)
print(f(20))

