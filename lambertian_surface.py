#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def rand_reflection(n=1):
    r = np.random.rand(np.int(n))
    return np.sqrt(r)


n = 1e+6
tau = rand_reflection(n)
plt.hist(tau, bins=25)
plt.xlabel(r"$\mu$")
plt.ylabel("#reflection direction hits")
plt.show()
