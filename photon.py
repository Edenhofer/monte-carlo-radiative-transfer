#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def rand_tau(n=1):
    r = np.random.rand(np.int(n))
    return -np.log(1 - r)


def rand_phi(n=1):
    return 2 * np.pi * np.random.rand(n)


def rand_mu(n=1, g=0.85):
    r = np.random.rand(np.int(n))
    return ((r * ((1 + g**2 - 2 * g)**(-1 / 2) - (1 + g**2 + 2 * g)**(-1 / 2)) + (1 + g**2 + 2 * g)**(-1 / 2))**-2 - 1 - g**2) / (-2 * g)


def rand_mu_reyleigh(n=1):
    r = np.random.rand(n)
    q = -8 * r + 4
    D = 1 + np.power(q, 2)/4
    u = np.power(-q/2 + np.sqrt(D), 1/3)
    return u - 1/u


def get_sca(p):
    pos_box = tuple(np.floor((p.pos * box_size) % beta_atm.shape).astype(int))

    tau = rand_tau(n=1)[0]
    phi = rand_phi(n =1)[0]
    if pos_box in clouds.keys():
        mu = rand_mu(n=1)[0]
    else:
        #mu = rand_mu_reyleigh(n=1)[0]
        mu = rand_mu(n=1)[0]

    #print(pos_box, p.pos)

    delta_s = tau / beta_atm[pos_box]

    return delta_s, mu, phi

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
low_photons = 100

box_size = (1., 1., 1.)
beta_atm = np.full((3, 3, 3), fill_value=1.)
clouds = {(0, 0, 1): 1.}
for el, key in clouds.items():
    beta_atm[el] = key

photon_counter = {"TOA": 0, "surface": 0}

if n_photons <= low_photons:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(-0.1, box_size[2]+0.1)

for i in range(n_photons):
    p = photon(0., 0., box_size[2], zenith_angle=0.)
    if n_photons <= low_photons:
        p_paths = [[p.pos[0]], [p.pos[1]], [p.pos[2]]]

    while p.pos[2] <= box_size[2] and p.pos[2] >= 0:
        # Propagate photon through a scattering atmopshere
        delta_s, mu, phi = get_sca(p)  # Incorporate clouds
        p.pos += p.n * delta_s

        u = np.array([1., 1., -(p.n[0] + p.n[1]) / p.n[2]])
        u /= np.linalg.norm(u)
        v = np.cross(p.n, u)
        # Adapt orientation after scattering
        p.n = mu * p.n + np.sin(np.arccos(mu)) * np.cos(phi) * u + np.sin(np.arccos(mu)) * np.sin(phi) * v
        p.n /= np.linalg.norm(p.n)

        if n_photons <= low_photons:
            print(p.pos, p.n)
            p_paths[0] += [p.pos[0]]
            p_paths[1] += [p.pos[1]]
            p_paths[2] += [p.pos[2]]

    if i % (n_photons/10) == 0 and n_photons > low_photons:
        print(i)

    if n_photons <= low_photons:
        ax.plot(p_paths[0], p_paths[1], p_paths[2])

    if p.pos[2] > box_size[2]:
        photon_counter["TOA"] += 1
    elif p.pos[2] < 0:
        photon_counter["surface"] += 1
    else:
        raise RuntimeError("photon did not pass through the atmosphere correctly!")

if n_photons <= low_photons:
    plt.show()

n_tot = photon_counter["surface"] + photon_counter["TOA"]
T = photon_counter["surface"] / (photon_counter["surface"] + photon_counter["TOA"])
T_std = np.sqrt((n_tot - photon_counter["surface"])/(n_tot * photon_counter["surface"]))
R = photon_counter["TOA"] / (photon_counter["surface"] + photon_counter["TOA"])
R_std = np.sqrt((n_tot - photon_counter["TOA"])/(n_tot * photon_counter["TOA"]))
assert n_tot == n_photons

print("T: {0:5.4f} ({1:5.4f});\tR: {2:5.4f} ({3:5.4f});\t{4}".format(T, T_std, R, R_std, photon_counter))
