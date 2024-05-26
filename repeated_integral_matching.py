import sympy as sp
from scipy.integrate import quad
from scipy.special import factorial
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import numpy as np

RADIUS = 5
N = 10 # because even integrals cancel all even terms and odd integrals odd, N must be an even number (or else there won't be enough equations to solve for all the even/all the odd terms)
f = lambda x: np.exp(-x**2)

# calculate the nth repeated integral of `f` across [-RADIUS, RADIUS] using Cauchy's repeated integration formula
def A(n):
    if n == 0: return f(RADIUS) - f(-RADIUS)
    return quad(lambda x: f(x) * (RADIUS - x) ** (n - 1) + f(-x) * (x - RADIUS) ** (n - 1), 0, RADIUS)[0] / factorial(n - 1)

x = sp.symbols("x")
coeffs = sp.symbols(f"c0:{N}")
P = sum(coeff * x**i for i, coeff in enumerate(coeffs))

# if `offset` > 0, skip over some of the first integrals; e.g. if `offset` is 2, f and its first integral are not set to equal those of the approximating function, but the `N`th and `N+1`th integrals are
offset = 0
for i in range(offset):
    P = sp.integrate(P, x)

# theoretically, you could match any of f's integrals and, as long as there are N equations, still get a valid system of linear equations to solve for coefficients; the integrals don't have to be sequential like this
eqs = []
for i in range(offset, N + offset):
    eqs.append(P.subs(x, RADIUS) - P.subs(x, -RADIUS) - A(i))
    if i < N + offset - 1:
        P = sp.integrate(P, x)

coeffs = list(sp.linsolve(eqs, coeffs))[0]
print(str.join(",", [f"{a:.20f}" for a in coeffs]))

x = np.linspace(-RADIUS, RADIUS, 1000)
y = f(x)
plt.plot(x, y, label="f(x)")
plt.plot(x, Polynomial(coeffs)(x), label="Integral-matched fit")
r = max(y) - min(y)
plt.ylim(min(y) - r * 0.1, max(y) + r * 0.1)
plt.legend()
plt.show()
