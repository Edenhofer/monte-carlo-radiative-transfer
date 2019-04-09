#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def rand_mu(n=1, g=0.85):
    r = np.random.rand(np.int(n))
    return ((r * ((1 + g**2 - 2*g)**(-1/2) - (1 + g**2 + 2*g)**(-1/2)) + (1 + g**2 + 2*g)**(-1/2))**-2 - 1 - g**2) / (-2 * g)


def henyey(g, mu):
    return (1 - np.power(g, 2)) / np.power(1 + np.power(g, 2) - 2 * g * mu, 3/2)


n = 100
mu = np.linspace(-1., 1., n)
for g in np.linspace(0.5, 1., 5):
    h = henyey(g, mu)
    plt.plot(mu, h, label="{0:.2f}".format(g))
plt.yscale("log")
plt.xlabel(r"$\mu$")
plt.ylabel("scattering phase function")
plt.legend()
plt.show()

n = 1e+6
mu = rand_mu(n, g=0.85)
plt.hist(mu, bins=25)
plt.xlabel(r"$\mu$")
plt.ylabel("#random hits")
plt.yscale("log")
plt.show()
