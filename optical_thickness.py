#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def rand_tau(n=1):
    r = np.random.rand(np.int(n))
    return - np.log(1 - r)


n = 1e+6
tau = rand_tau(n)
plt.hist(tau, bins=25)
plt.xlabel(r"$\tau$")
plt.ylabel("#random hits")
plt.yscale("log")
plt.show()
