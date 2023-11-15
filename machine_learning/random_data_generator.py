import numpy as np


def generate(length):
    size = 972
    gen = np.random.default_rng(np.random.randint(0, size, 1))
    gyro = [list(gen.normal(0, 3, 3))]
    acc = [list(gen.normal(0, 3, 3))]
    x = [gen.integers(0, size)]
    y = [gen.integers(0, size)]
    for i in range(length - 1):
        accn = list(gen.normal(0, 3, 3))
        acc.append(accn)
        gyron = list(gen.normal(0, 0.01, 3))
        gyro.append(gyron)

        xn = -1
        while not 0 < xn < size:
            xn = x[i] + gen.uniform(-3, 3)
        x.append(xn)
        yn = -1
        while not 0 < yn < size:
            yn = y[i] + gen.uniform(-3, 3)
        y.append(yn)
    time = np.random.randint(1000000, 3000000, length)
    return dict(
        (
            ("size", size),
            ("gyro", gyro),
            ("acc", acc),
            ("x", x),
            ("y", y),
            ("time", time),
        )
    )
