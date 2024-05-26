import numpy as np
import matplotlib.pyplot as plt

RADIUS = 5
N = 10
f = lambda x: np.exp(-x**2)

x = np.linspace(-RADIUS, RADIUS, 1000)
y = f(x)

poly = np.polynomial.Polynomial.fit(x, y, N - 1)
print(str.join(",", [f"{c:.20f}" for c in poly.convert().coef]))

plt.plot(x, y, label="f(x)")
plt.plot(x, poly(x), label="Least-squares fit")
r = max(y) - min(y)
plt.ylim(min(y) - r * 0.1, max(y) + r * 0.1)
plt.legend()
plt.show()
