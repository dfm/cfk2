# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = []

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from scipy.linalg import cho_solve, cho_factor


class K2PSF(object):

    def __init__(self, mu, cov):
        self.mu1 = mu
        self.mu2 = mu
        self.cov1 = np.array(cov)
        cov[np.diag_indices_from(cov)] *= 4
        self.cov2 = cov

        self.alpha = 0.1

    @property
    def cov1(self):
        return self._cov1

    @cov1.setter
    def cov1(self, cov):
        self._cov1 = cov
        self._factor1 = cho_factor(cov)
        self._lndet1 = 2 * np.sum(np.log(np.diag(self._factor1[0])))

    @property
    def cov2(self):
        return self._cov2

    @cov2.setter
    def cov2(self, cov):
        self._cov2 = cov
        self._factor2 = cho_factor(cov)
        self._lndet2 = 2 * np.sum(np.log(np.diag(self._factor2[0])))

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, a):
        self._alpha = a
        self.mixture = np.array([1. - a, a])

    def __call__(self, coords):
        r1 = (coords - self.mu1).T
        m1 = np.exp(-0.5*(np.sum(r1 * cho_solve(self._factor1, r1), axis=0)
                          + self._lndet1))
        r2 = (coords - self.mu2).T
        m2 = np.exp(-0.5*(np.sum(r2 * cho_solve(self._factor2, r2), axis=0)
                          + self._lndet2))
        return self.mixture[0] * m1 + self.mixture[1] * m2

    def bounds(self):
        return [
            (-1., 1.), (-1., 1.),
            (-4., 1.), (-4., 1.),
            (-0.5, 0.5),
            (0.1, 0.5),
            (-1., 1.), (-1., 1.),
            (-4., 1.), (-4., 1.),
            (-0.5, 0.5),
        ]

    @property
    def vector(self):
        return np.concatenate((self.mu1, np.log(np.diag(self.cov1)),
                               [self.cov1[1, 0], self.alpha],
                               self.mu2, np.log(np.diag(self.cov2)),
                               [self.cov2[1, 0]]))

    @vector.setter
    def vector(self, v):
        self.mu1 = v[:2]
        c = np.diag(np.exp(v[2:4]))
        c[1, 0] = v[4]
        c[0, 1] = v[4]
        self.cov1 = c

        self.alpha = v[5]
        self.mu2 = v[6:8]
        c = np.diag(np.exp(v[8:10]))
        c[1, 0] = v[10]
        c[0, 1] = v[10]
        self.cov2 = c


class K2Stack(object):

    def __init__(self, frames, psf, sources, bkg_order=3):
        self.frames = frames
        self.psf = psf
        self.sources = sources

        shape = self.frames.shape[1:]
        x, y = np.meshgrid(np.arange(shape[0]),
                           np.arange(shape[1]),
                           indexing="ij")
        self.delta = (self.sources[:, None, :] -
                      (np.vstack((x.flatten(), y.flatten())).T)[None, :, :])

        self.A = np.concatenate((
            np.empty(self.delta.shape[:2]),
            np.vander(x.flatten(), bkg_order+1)[:, :-1].T,
            np.vander(y.flatten(), bkg_order+1).T
        ), axis=0)

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
        n = len(self.sources)
        bkg = np.dot(self.light_curves[ind, n:],
                     self.A[n:]).reshape(data.shape)
        ax.imshow(bkg.T, cmap="gray", interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("background")

        # Plot the PSF.
        ax = axes[1, 2]
        x, y = np.meshgrid(np.arange(-10, 10),
                           np.arange(-10, 10),
                           indexing="ij")
        c = np.vstack((x.flatten(), y.flatten())).T
        ax.imshow(self.psf(c).reshape(x.shape), cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("psf")

        return fig

    def _do_convolution(self):
        for i, r in enumerate(self.delta):
            self.A[i] = self.psf(r)

    def update_light_curves(self):
        # First convolve by the PSF.
        self._do_convolution()
        ATA = cho_factor(np.dot(self.A, self.A.T), overwrite_a=True)

        # Loop over the frames and apply the solve.
        for i, img in enumerate(self.frames):
            ATy = np.dot(self.A, img.flatten())
            self.light_curves[i] = cho_solve(ATA, ATy, overwrite_b=True)

        n = len(self.sources)
        t = self.light_curves[:, :n]
        t[t < 0.0] = 0.0
        self.light_curves[:, :n] = t

    def chi2(self, p):
        try:
            self.psf.vector = p
        except np.linalg.LinAlgError:
            return 1e15
        self.update_light_curves()

        r2 = 0.0
        for i, w in enumerate(self.light_curves):
            r2 += np.mean((self.frames[i].flatten() - np.dot(w, self.A))**2)
        c2 = r2 / len(self.light_curves)
        print(p, c2)
        return c2

    def optimize(self):
        r = op.minimize(self.chi2, self.psf.vector, method="L-BFGS-B",
                        bounds=self.psf.bounds())
        self.psf.vector = r.x
        self.update_light_curves()
        print(r)
