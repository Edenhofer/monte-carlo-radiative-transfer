#!/usr/bin/env python3

import numpy as np
import itertools
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
    D = 1 + np.power(q, 2) / 4
    u = np.power(-q / 2 + np.sqrt(D), 1 / 3)
    return u - 1 / u


def get_planes(pos_box, ni_dir, direction):
    b = np.array(pos_box)
    if ni_dir < 0:
        pl_0 = b * atm_size / beta_sca.shape
        a = np.zeros(3).astype(int)
        a[(direction + 1) % 3] = 1
        pl_1 = (b + a) * atm_size / beta_sca.shape
        a = np.zeros(3).astype(int)
        a[(direction + 2) % 3] = 1
        pl_2 = (b + a) * atm_size / beta_sca.shape
    else:
        offset = np.zeros(3).astype(int)
        offset[direction] += 1
        pl_0 = (b + offset) * atm_size / beta_sca.shape
        a = np.zeros(3).astype(int)
        a[(direction + 1) % 3] = 1
        pl_1 = (b + offset + a) * atm_size / beta_sca.shape
        a = np.zeros(3).astype(int)
        a[(direction + 2) % 3] = 1
        pl_2 = (b + offset + a) * atm_size / beta_sca.shape

    if direction == 0:
        pl_0[2] = atm_height[b[2]]
        pl_1[2] = atm_height[b[2]]
        pl_2[2] = atm_height[b[2]+1]
    elif direction == 1:
        pl_0[2] = atm_height[b[2]]
        pl_1[2] = atm_height[b[2]+1]
        pl_2[2] = atm_height[b[2]]
    elif direction == 2:
        if ni_dir < 0:
            pl_0[2] = atm_height[b[2]]
            pl_1[2] = atm_height[b[2]]
            pl_2[2] = atm_height[b[2]]
        else:
            if b[2]+1 == atm_height.shape[0]:
                raise RuntimeError("there can not be a plane for this box")

            pl_0[2] = atm_height[b[2]+1]
            pl_1[2] = atm_height[b[2]+1]
            pl_2[2] = atm_height[b[2]+1]

    return pl_0, pl_1, pl_2


def get_box(pos):
    idx = np.where(atm_height < pos[2])[0]
    if idx.shape == (0,):
        return (None, None, None)

    box = np.floor((pos * beta_sca.shape / atm_size) % beta_sca.shape).astype(int)
    box[2] = idx.max()
    return tuple(box)


def get_sca(p):
    tau = rand_tau(n=1)[0]
    phi = rand_phi(n=1)[0]
    # NOTE: adapting the position to modulo atm_size leads to wrong distances
    if get_box(p.pos) in clouds.keys():
        mu = rand_mu(n=1)[0]
    else:
        mu = rand_mu_reyleigh(n=1)[0]

    delta_s = 0
    delta_tau = 0
    p_prop_pos = p.pos + p.n * delta_s
    p_prop_box = get_box(p_prop_pos)
    while delta_tau < tau and p_prop_pos[2] < atm_size[2] and p_prop_pos[2] > 0.:
        # Calculate intersecting boxes starting with the box below
        for i in range(2, -1, -1):
            pl_0, pl_1, pl_2 = get_planes(p_prop_box, p.n[i], i)

            # Calculate the intersection between the photon and the current box
            pl_01 = pl_1 - pl_0
            pl_02 = pl_2 - pl_0
            t = (np.cross(pl_01, pl_02).dot((p_prop_pos % atm_size) - pl_0)) / (-(p.n).dot(np.cross(pl_01, pl_02)))
            pos_int = (p_prop_pos % atm_size) + p.n * t

            if pos_int[i] is np.nan and i == 2:
                raise RuntimeError("no intersection found")
            elif pos_int[i] is np.nan:
                continue

            left_bound_first_orthogonal = min(pl_0[(i + 1) % 3], pl_1[(i + 1) % 3], pl_2[(i + 1) % 3])
            right_bound_first_orthogonal = max(pl_0[(i + 1) % 3], pl_1[(i + 1) % 3], pl_2[(i + 1) % 3])
            left_bound_second_orthogonal = min(pl_0[(i + 2) % 3], pl_1[(i + 2) % 3], pl_2[(i + 2) % 3])
            right_bound_second_orthogonal = max(pl_0[(i + 2) % 3], pl_1[(i + 2) % 3], pl_2[(i + 2) % 3])
            # Reduce the position implicitly to the size of the whole atmosphere
            pos_int_mod = pos_int % atm_size
            if pos_int_mod[(i + 1) % 3] >= left_bound_first_orthogonal and pos_int_mod[(i + 1) % 3] <= right_bound_first_orthogonal \
                and pos_int_mod[(i + 2) % 3] >= left_bound_second_orthogonal and pos_int_mod[(i + 2) % 3] <= right_bound_second_orthogonal:

                break
            elif i == 0:
                raise RuntimeError("no intersecting box found")

        # Move a little bit further as to always cross into the next box
        l = np.linalg.norm(pos_int - (p_prop_pos % atm_size)) + 1e-7
        tau_box = l * beta_sca[tuple(p_prop_box)]
        tau_step = min(tau - delta_tau, tau_box)
        delta_tau += tau_step
        # Skip the box if there is absolutely no scattering
        if beta_sca[tuple(p_prop_box)] == 0.:
            s_step = l
        else:
            s_step = tau_step / beta_sca[tuple(p_prop_box)]

        p.w_ln -= beta_abs[tuple(p_prop_box)] * s_step
        delta_s += s_step
        # Adapt propagation position for next iteration
        p_prop_pos = p.pos + p.n * delta_s
        p_prop_box = get_box(p_prop_pos)

        #print("p.pos: {pos}; p.n: {dir}; int: {int}; l: {l}; tau: {tau:5.4f}; delta_tau: {delta_tau:5.4f}; beta_sca: {beta_sca}".format(pos=p.pos, dir=p.n, int=pos_int, l=l, tau=tau, delta_tau=delta_tau, beta_sca=beta_sca[tuple(p_prop_box)]))

    return delta_s, mu, phi


class photon(object):
    def __init__(self, x=None, y=None, z=None, n_x=None, n_y=None, n_z=None, zenith_angle=None, weight=None):
        x = 0. if x is None else x
        y = 0. if y is None else y
        z = 0. if z is None else z
        self.pos = np.array([x, y, z])

        self.w_ln = 0. if weight is None else np.log(weight)
        self.w = None if weight is None else weight

        n_x = 0. if n_x is None else n_x
        n_y = 0. if n_y is None else n_y
        n_z = -1. if n_z is None else n_z
        if zenith_angle is not None:
            n_x = np.sin(zenith_angle)
            n_y = 0
            n_z = -np.cos(zenith_angle)

        self.n = np.array([n_x, n_y, n_z])

    def weight(self):
        return np.exp(self.w_ln)


n_photons = int(1e+2)
zenith_angle = 0.
low_photons = 100

atm_tab = np.loadtxt("ksca_kabs/lambda320.dat")
h = atm_tab[:, 0].max() - atm_tab[:, 0].min()
atm_size = (2. * h, 2. * h, h)
atm_height = atm_tab[:, 0][::-1]
beta_sca = np.ones((3, 3, atm_tab.shape[0]))
beta_sca *= atm_tab[:, 1][::-1]
beta_abs = np.ones((3, 3, atm_tab.shape[0]))
beta_abs *= atm_tab[:, 2][::-1]

clouds = {}
#clouds = {(1, 1, int(atm_tab.shape[0] / 2.)): 10.}
for el, key in clouds.items():
    beta_sca[el] = key

photon_counter = {"TOA": 0, "TOA_weight": 0., "surface": 0, "surface_weight": 0.}

if n_photons <= low_photons:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.1, atm_size[0] + 0.1)
    ax.set_ylim(-0.1, atm_size[1] + 0.1)
    ax.set_zlim(-0.1, atm_size[2] + 0.1)

for i in range(n_photons):
    p = photon(atm_size[0] / 2., atm_size[1] / 2., atm_size[2] - 1e-5, zenith_angle=zenith_angle)
    if n_photons <= low_photons:
        p_paths = [[p.pos[0]], [p.pos[1]], [p.pos[2]]]

    while p.pos[2] < atm_size[2] and p.pos[2] > 0:
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
            p_paths[0] += [p.pos[0]]
            p_paths[1] += [p.pos[1]]
            p_paths[2] += [p.pos[2]]

    if i % (n_photons / 10) == 0 and n_photons > low_photons:
        print(i)

    if n_photons <= low_photons:
        print("{i:3d} :: {pos} {n}".format(i=i, pos=p.pos, n=p.n))
        ax.plot(p_paths[0], p_paths[1], p_paths[2])

    if p.pos[2] >= atm_size[2]:
        photon_counter["TOA"] += 1
        photon_counter["TOA_weight"] += p.weight()
    elif p.pos[2] <= 0:
        photon_counter["surface"] += 1
        photon_counter["surface_weight"] += p.weight()
    else:
        raise RuntimeError("photon did not pass through the atmosphere correctly")

n_tot = photon_counter["surface"] + photon_counter["TOA"]
assert n_tot == n_photons

T = photon_counter["surface_weight"] / n_tot
try:
    T_std = np.sqrt((n_tot - photon_counter["surface_weight"]) / (n_tot * photon_counter["surface_weight"]))
except ZeroDivisionError:
    T_std = np.inf

R = photon_counter["TOA_weight"] / n_tot
try:
    R_std = np.sqrt((n_tot - photon_counter["TOA_weight"]) / (n_tot * photon_counter["TOA_weight"]))
except ZeroDivisionError:
    R_std = np.inf

print("T: {0:5.4f} ({1:5.4f});\tR: {2:5.4f} ({3:5.4f});\t{4}".format(T, T_std, R, R_std, photon_counter))

if n_photons <= low_photons:
    plt.show()
