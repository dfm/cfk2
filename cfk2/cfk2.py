# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np
import matplotlib.pyplot as pl
from scipy.linalg import cho_solve, cho_factor

from .utils import convolution_indices


class K2Stack(object):

    def __init__(self, frames, psf, sources, bkg_order=3, sup=2):
        self.frames = frames
        self.psf = psf / np.sum(psf)
        self.sources = sources

        # Build the base image.
        base_shape = np.array(self.frames.shape[1:])
        shape = sup * base_shape + np.array(self.psf.shape) - 1
        self.base_images = np.zeros((len(self.sources), shape[0], shape[1]),
                                    dtype=float)
        x, y = np.meshgrid(np.arange(shape[0])/float(sup),
                           np.arange(shape[1])/float(sup),
                           indexing="ij")
        self.psf_offset = (self.psf.shape[0] // 2,
                           self.psf.shape[1] // 2)
        x -= self.psf_offset[0] / float(sup)
        y -= self.psf_offset[1] / float(sup)
        for i, (xi, yi) in enumerate(self.sources):
            r2 = (x - xi)**2 + (y - yi)**2
            self.base_images[i] = np.exp(-0.5 * r2 / 0.5**2)
            # r = np.sqrt((x - xi)**2 + (y - yi)**2)
            # m = r < 1.0
            # self.base_images[i, m] = r[m] / r[m].sum()

        # Build the design matrix.
        x, y = np.meshgrid(np.arange(base_shape[0]), np.arange(base_shape[1]),
                           indexing="ij")
        self.A = np.concatenate((
            np.empty((len(self.base_images), np.prod(base_shape))),
            np.vander(x.flatten(), bkg_order+1)[:, :-1].T,
            np.vander(y.flatten(), bkg_order+1).T
        ), axis=0)

        # Get the convolution indices.
        self.inds = convolution_indices(shape, self.psf.shape, sup=sup)

        # Initialize the light curves.
        self.light_curves = np.empty((len(self.frames), self.A.shape[0]))

    def plot_frame(self, ind):
        fig, axes = pl.subplots(2, 3, figsize=(15, 10))

        w = self.light_curves[ind]
        data = self.frames[ind]
        mu = np.mean(data)
        std = np.std(data)
        vrng = (mu-std, mu+2*std)

        # Plot the data.
        ax = axes[0, 0]
        ax.imshow(data.T, vmin=vrng[0], vmax=vrng[1], cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("data")

        # # Plot the base image.
        # ax = axes[1, 0]
        # img = np.dot(w[:len(self.base_images)],
        #              self._get_base(flat=True)).reshape(data.shape)
        # mu = np.mean(img)
        # std = np.std(img)
        # ax.imshow(img.T, vmin=mu-std, vmax=mu+std, cmap="gray",
        #           interpolation="nearest")
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_title("base scene")

        # Plot the model.
        ax = axes[0, 1]
        self._do_convolution()
        img = np.dot(w, self.A).reshape(data.shape)
        ax.imshow(img.T, vmin=vrng[0], vmax=vrng[1], cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("model")

        # Plot the residuals.
        ax = axes[0, 2]
        w = vrng[1] - vrng[0]
        ax.imshow((data - img).T, vmin=-w, vmax=w, cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("residuals")

        # Plot the background.
        ax = axes[1, 1]
        n = len(self.base_images)
        bkg = np.dot(self.light_curves[ind, n:],
                     self.A[n:]).reshape(data.shape)
        ax.imshow(bkg.T, cmap="gray", interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("background")

        # Plot the PSF.
        ax = axes[1, 2]
        ax.imshow(self.psf, cmap="gray", interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("psf")

        return fig

    def _get_base(self, flat=False):
        n = len(self.base_images)
        o = self.psf_offset
        s = self.base_images.shape[1:]
        bi = self.base_images[:, o[0]:o[0]+s[0], o[1]:o[1]+s[1]]
        if flat:
            return bi.reshape(n, -1)
        return bi

    def _do_convolution(self):
        flat_psf = self.psf.flatten()
        for i, img in enumerate(self.base_images):
            self.A[i] = np.dot(img[self.inds], flat_psf)

    def update_light_curves(self):
        # First convolve by the PSF.
        self._do_convolution()
        ATA = cho_factor(np.dot(self.A, self.A.T), overwrite_a=True)

        # Loop over the frames and apply the solve.
        for i, img in enumerate(self.frames):
            ATy = np.dot(self.A, img.flatten())
            self.light_curves[i] = cho_solve(ATA, ATy, overwrite_b=True)

        n = len(self.base_images)
        t = self.light_curves[:, :n]
        t[t < 0.0] = 0.0
        self.light_curves[:, :n] = t

        return self.light_curves[:, :n], self.light_curves[:, n:]

    def update_psf(self):
        n = len(self.base_images)
        bkg = np.dot(self.light_curves[:, n:], self.A[n:])
        y = self.frames.reshape(len(self.frames), -1) - bkg

        psf = np.zeros(np.prod(self.psf.shape))
        for i, w in enumerate(self.light_curves):
            img = np.sum(w[:n, None, None] * self.base_images, axis=0)
            B = img[self.inds]
            BT = B.T
            psf += np.linalg.solve(np.dot(BT, B), np.dot(BT, y[i]))
        self.psf = psf.reshape(self.psf.shape)
        self.psf[self.psf < 0.0] = 0.0
        self.psf /= np.sum(self.psf)
        return self.psf
