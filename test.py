#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import h5py
import fitsio
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as pl

from cfk2.cfk2 import K2Stack, GaussianPSF, QuadraticPSF, MixturePSF


xrng = (200, 301)
yrng = (400, 500)
with h5py.File("image.h5", "r") as f:
    frames = f["frames"][-100:, xrng[0]:xrng[1], yrng[0]:yrng[1]]

# Load the WCS.
hdr = fitsio.read_header("wcs.fits")
wcs = WCS(hdr)

# Load the 2MASS catalog.
tm_ra, tm_dec = np.loadtxt("2mass.txt", skiprows=103, usecols=(0, 1),
                           unpack=True)
tm_y, tm_x = wcs.wcs_world2pix(tm_ra, tm_dec, 0)

# Only take sources in the range.
m = (tm_x > xrng[0]-1) & (tm_x < xrng[1]+2)
m &= (tm_y > yrng[0]-1) & (tm_y < yrng[1]+2)

# Correct the coordinates.
tm_x = tm_x[m] - xrng[0] - 0.5
tm_y = tm_y[m] - yrng[0]
coords = np.array((tm_x, tm_y)).T

c = np.diag([0.5, 0.5])
c[1, 0] = c[0, 1] = 1e-3
mu = np.array([0.0, 0.0])
psf = GaussianPSF(mu, c)
# psf = MixturePSF(
#     0.5,
#     GaussianPSF(mu, c),
#     GaussianPSF(mu, c * 2.0),
# )

stack = K2Stack(frames, psf, coords)

print(stack.chi2(stack.psf.vector))

# stack.update_light_curves()
stack.optimize()
fig = stack.plot_frame(-1)
fig.savefig("frame.png", bbox_inches="tight")

# for i in range(5):
#     print(i)
#     stack.update_light_curves()
#     stack.update_psf()
#     fig = stack.plot_frame(-1)
#     fig.savefig("frames/{0:03d}.png".format(i), bbox_inches="tight")
#     pl.close(fig)
#     break
