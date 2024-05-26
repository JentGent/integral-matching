import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.misc import derivative
from numpy.polynomial.polynomial import Polynomial

RADIUS = 5
N = 10 # the derivatives blow up, so spacing needs to be high and N low
SPACING = 0.1
f = lambda x: np.exp(-x**2)

x = np.linspace(-RADIUS, RADIUS, 1000)
y = f(x)

coeffs = [derivative(f, 0, SPACING, i, order=2 * i + 1) / factorial(i) for i in range(N)]
print(str.join(",", [f"{c:.20f}" for c in coeffs]))

plt.plot(x, y, label="f(x)")
plt.plot(x, Polynomial(coeffs)(x), label="Taylor polynomial")
r = max(y) - min(y)
plt.ylim(min(y) - r * 0.1, max(y) + r * 0.1)
plt.legend()
plt.show()