#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def rand_tau(n=1):
    r = np.random.rand(np.int(n))
    return -np.log(1 - r)


def rand_mu(n=1, g=0.85):
    r = np.random.rand(np.int(n))
    return ((r * ((1 + g**2 - 2 * g)**(-1 / 2) - (1 + g**2 + 2 * g)**(-1 / 2)) + (1 + g**2 + 2 * g)**(-1 / 2))**-2 - 1 - g**2) / (-2 * g)


class photon(object):
    def __init__(self, x=None, y=None, z=None, n_r=None, n_mu=None, n_phi=None):
        self.x = 0. if x is None else x
        self.y = 0. if y is None else y
        self.z = 0. if z is None else z

        self.n_r = 1. if n_r is None else n_r
        self.n_mu = 0. if n_mu is None else n_mu
        self.n_phi = 0. if n_phi is None else n_phi

    def direction(self):
        return np.array([self.n_r, self.n_mu, self.n_phi])

    def pos(self):
        return np.array([self.x, self.y, self.z])


n_photons = int(1e+3)
z_TOA = 1.
beta_ext = 1e-0

photon_counter = {"toa": 0, "surface": 0}

for i in range(n_photons):
    p = photon(0., 0., z_TOA)
    while p.z <= z_TOA and p.z >= 0:
        tau = rand_tau(n=1)[0]
        delta_s = tau / beta_ext
        p.x += np.sin(np.arccos(p.n_mu)) * np.cos(p.n_phi) * delta_s
        p.y += np.sin(np.arccos(p.n_mu)) * np.sin(p.n_phi) * delta_s
        p.z -= p.n_mu * delta_s

        mu = rand_mu(n=1)[0]
        phi = 2 * np.pi * np.random.rand(1)[0]

        p.n_mu += mu

        # Ensure proper boundaries
        if p.n_mu > 1.:
            p.n_mu = -1. * (p.n_mu % 1)
        elif p.n_mu < -1.:
            p.n_mu = p.n_mu % 1

        p.n_phi += phi
        p.n_phi = p.n_phi % (2 * np.pi)

        print(p.pos(), p.direction())

    if i % 100 == 0:
        print(i)

    if p.z > z_TOA:
        photon_counter["toa"] += 1
    elif p.z < 0:
        photon_counter["surface"] += 1
    else:
        raise RuntimeError("BAD BAD BAD!")

print(photon_counter)
