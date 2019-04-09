#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

n = int(1e+6)
r = np.random.rand(n)
q = -8 * r + 4
D = 1 + np.power(q, 2)/4
u = np.power(-q/2 + np.sqrt(D), 1/3)
mu = u - 1/u
print(mu)

plt.hist(mu, bins=25, density=True)
plt.xlabel(r"$\mu$")
plt.ylabel("normalized #random hits")
plt.show()
