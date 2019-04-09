#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def rand_tau(n=1):
    r = np.random.rand(np.int(n))
    return -np.log(1 - r)


def rand_phi(n=1):
    return 2 * np.pi * np.random.rand(n)


def rand_mu(n=1, g=0.85):
    r = np.random.rand(np.int(n))
    return ((r * ((1 + g**2 - 2 * g)**(-1 / 2) - (1 + g**2 + 2 * g)**(-1 / 2)) + (1 + g**2 + 2 * g)**(-1 / 2))**-2 - 1 - g**2) / (-2 * g)


class photon(object):
    def __init__(self, x=None, y=None, z=None, n_x=None, n_y=None, n_z=None, zenith_angle=None):
        x = 0. if x is None else x
        y = 0. if y is None else y
        z = 0. if z is None else z
        self.pos = np.array([x, y, z])

        n_x = 0. if n_x is None else n_x
        n_y = 0. if n_y is None else n_y
        n_z = -1. if n_z is None else n_z
        if zenith_angle is not None:
            n_x = np.sin(zenith_angle)
            n_y = 0
            n_z = -np.cos(zenith_angle)

        self.n = np.array([n_x, n_y, n_z])


n_photons = int(1e+4)
z_TOA = 1.
beta_ext = 1e+0

photon_counter = {"TOA": 0, "surface": 0}

for i in range(n_photons):
    p = photon(0., 0., z_TOA, zenith_angle=np.pi / 6.)
    while p.pos[2] <= z_TOA and p.pos[2] >= 0:
        tau = rand_tau(n=1)[0]
        delta_s = tau / beta_ext
        # Propagate photon
        p.pos += p.n * delta_s

        u = np.array([1., 1., -(p.n[0] + p.n[1]) / p.n[2]])
        u /= np.linalg.norm(u)
        v = np.cross(p.n, u)

        # Scatter photon
        mu = rand_mu(n=1)[0]
        phi = rand_phi(n=1)[0]
        p.n = mu * p.n + np.sin(np.arccos(mu)) * np.cos(phi) * u + np.sin(np.arccos(mu)) * np.sin(phi) * v
        p.n /= np.linalg.norm(p.n)

        if n_photons < 10:
            print(p.pos, p.n)

    if i % (n_photons/10) == 0:
        print(i)

    if p.pos[2] > z_TOA:
        photon_counter["TOA"] += 1
    elif p.pos[2] < 0:
        photon_counter["surface"] += 1
    else:
        raise RuntimeError("photon did not pass through the atmosphere correctly!")

T = photon_counter["surface"] / (photon_counter["surface"] + photon_counter["TOA"])
R = photon_counter["TOA"] / (photon_counter["surface"] + photon_counter["TOA"])
print("T: {0:5.4f};\tR: {1:5.4f};\t{2}".format(T, R, photon_counter))
