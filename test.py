#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import h5py
import fitsio
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as pl

from cfk2.cfk2 import K2Stack, K2PSF


xrng = (300, 353)
yrng = (650, 700)
with h5py.File("image.h5", "r") as f:
    frames = f["frames"][-500:, xrng[0]:xrng[1], yrng[0]:yrng[1]]

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

# psf_hw = 10
# px, py = np.meshgrid(range(-psf_hw, psf_hw+1), range(-psf_hw-1, psf_hw+2),
#                      indexing="ij")
# psf = np.exp(-0.5*(px**2 + py**2) / 1.0**2)
psf = K2PSF(np.array([0.2, -0.05]), np.diag(np.exp([-1.123, -1.2098])))

stack = K2Stack(frames, psf, coords)
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
