# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["GaussianPSF", "K2Stack"]

import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as pl
from scipy.linalg import cho_solve, cho_factor


class MixturePSF(object):

    def __init__(self, alpha, psf1, psf2):
        self.alpha = alpha
        self.psf1 = psf1
        self.psf2 = psf2

    def __len__(self):
        return 1 + len(self.psf1) + len(self.psf2)

    def bounds(self):
        return np.concatenate(([[0.01, 0.99]],
                               self.psf1.bounds(),
                               self.psf2.bounds()))

    @property
    def vector(self):
        return np.concatenate(([self.alpha],
                               self.psf1.vector,
                               self.psf2.vector))

    @vector.setter
    def vector(self, v):
        self.alpha = v[0]
        n = len(self.psf1) + 1
        self.psf1.vector = v[1:n]
        self.psf2.vector = v[n:]

    def __call__(self, *args):
        a = self.psf1(*args)
        b = self.psf2(*args)
        return self.alpha * a + (1 - self.alpha) * b


class QuadraticPSF(object):

    def __init__(self, mu, width):
        self.mu = mu
        self.width = width

    def __len__(self):
        return 3

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, w):
        self._width = w
        self._w2 = w**2
        self._a = 0.75 / (self._w2 * w)

    def __call__(self, coords):
        r2 = (coords - self.mu)**2
        m = r2[:, 0] < 0.25**2
        r = np.zeros(len(coords))
        r[m] = self._a * (self._w2 - r2[m, 1])
        r[r < 0.0] = 0.0
        return r

    def bounds(self):
        return [
            (-0.5, 0.5), (-0.5, 0.5),
            (np.log(0.5), np.log(5.)),
        ]

    @property
    def vector(self):
        return np.concatenate((self.mu, [np.log(self.width)]))

    @vector.setter
    def vector(self, v):
        self.mu = v[:2]
        self.width = np.exp(v[2])


class GaussianPSF(object):

    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = np.array(cov)

    def __len__(self):
        return 5

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, cov):
        self._cov = cov
        d = np.diag(cov)
        det = np.prod(d) - cov[1, 0]**2
        if det <= 0.0:
            raise ValueError()
        self._factor = 1.0 / np.sqrt(2 * np.pi * det)
        self._inv_cov = np.diag(d[::-1] / det)
        self._inv_cov[1, 0] = self._inv_cov[0, 1] = -cov[1, 0] / det

    def __call__(self, coords):
        r = (coords - self.mu).T
        return self._factor * np.exp(-0.5*np.sum(r * self._inv_cov.dot(r),
                                                 axis=0))

    def bounds(self):
        return [
            (-1., 1.), (-1., 1.),
            (-4., 1.), (-4., 1.),
            (-2, 2),
        ]

    @property
    def vector(self):
        return np.concatenate((self.mu, np.log(np.diag(self.cov)),
                               [self.cov[1, 0]]))

    @vector.setter
    def vector(self, v):
        self.mu = v[:2]
        c = np.diag(np.exp(v[2:4]))
        c[1, 0] = v[4]
        c[0, 1] = v[4]
        self.cov = c


class K2Stack(object):

    def __init__(self, frames, psf, sources):
        self.frames = frames
        self.psf = psf
        self.sources = sources

        shape = self.frames.shape[1:]
        x, y = np.meshgrid(np.arange(shape[0]),
                           np.arange(shape[1]),
                           indexing="ij")
        self.delta = (self.sources[:, None, :] -
                      (np.vstack((x.flatten(), y.flatten())).T)[None, :, :])

        x, y = x.flatten(), y.flatten()
        self.A = np.concatenate((
            np.empty(self.delta.shape[:2]),
            np.ones((1, len(x))),
            # [(x == i).astype(float) for i in np.arange(shape[0])],
        ), axis=0)

        # Compute the column-wise background model.
        self.bkg = np.zeros_like(self.frames)
        self.bkg += np.median(self.frames, axis=2)[:, :, None]
        self._bkg_sub = self.frames - self.bkg
        self.bkg = self.bkg.reshape((len(self.frames), -1))

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

        # Plot the fixed background image.
        ax = axes[1, 0]
        ax.imshow(self.bkg[ind].reshape(data.shape).T, cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("fixed background")

        # Plot the model.
        ax = axes[0, 1]
        self._do_convolution()
        img = (np.dot(w, self.A) + self.bkg[ind]).reshape(data.shape)
        ax.imshow(img.T, vmin=vrng[0], vmax=vrng[1], cmap="gray",
                  interpolation="nearest")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title("model")

        # Plot the residuals.
        ax = axes[0, 2]
        w = vrng[1] - vrng[0]
        ax.imshow((data - img).T, vmin=-w, vmax=w,
                  cmap="gray", interpolation="nearest")
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
        x, y = np.meshgrid(np.arange(-10, 10, 0.1),
                           np.arange(-10, 10, 0.1),
                           indexing="ij")
        c = np.vstack((x.flatten(), y.flatten())).T
        ax.imshow(self.psf(c).reshape(x.shape).T, cmap="gray",
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
        for i, img in enumerate(self._bkg_sub):
            ATy = np.dot(self.A, img.flatten())
            self.light_curves[i] = cho_solve(ATA, ATy, overwrite_b=True)

        # n = len(self.sources)
        # t = self.light_curves[:, :n]
        # t[t < 0.0] = 0.0
        # self.light_curves[:, :n] = t

    def chi2(self, p):
        try:
            self.psf.vector = p
        except (ValueError, np.linalg.LinAlgError):
            return 1e15
        self.update_light_curves()

        r2 = 0.0
        for i, w in enumerate(self.light_curves):
            img = self.frames[i].flatten()
            mod = np.dot(w, self.A) + self.bkg[i]
            r2 += np.mean((img - mod)**2)  # / mod)
        c2 = r2 / len(self.light_curves)
        print(p, c2)
        return c2

    def optimize(self):
        r = op.minimize(self.chi2, self.psf.vector, method="L-BFGS-B",
                        bounds=self.psf.bounds())
        self.psf.vector = r.x
        self.update_light_curves()
        print(r)
