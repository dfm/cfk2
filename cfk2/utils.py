# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["convolution_indices"]

import numpy as np


def convolution_indices(shape1, shape2, sup=2):
    s1 = np.array(shape1)
    s2 = np.array(shape2)

    x1, y1 = (_.flatten() for _ in np.meshgrid(np.arange(0, s1[0], sup),
                                               np.arange(0, s1[1], sup),
                                               indexing="ij"))
    x2, y2 = (_.flatten() for _ in np.meshgrid(np.arange(s2[0]),
                                               np.arange(s2[1]),
                                               indexing="ij"))
    x = x1[:, None] + x2[None, :]
    y = y1[:, None] + y2[None, :]

    # "Valid" range.
    m = (x >= 0) & (x < s1[0])
    m &= (y >= 0) & (y < s1[1])
    m = np.all(m, axis=1)
    x, y = x[m], y[m]
    return x, y


if __name__ == "__main__":
    from scipy.signal import convolve

    np.random.seed(1234)
    img1 = np.arange(10 * 12).reshape(10, 12)
    img2 = np.arange(2 * 4).reshape(2, 4)
    print("img 1")
    print(img1)
    print("img 2")
    print(img2)

    # Do the FFT convolution.
    fft_res = convolve(img1, img2, mode="valid")

    # Do the linear algebra convolution.
    x, y = convolution_indices(img1.shape, img2.shape, sup=4)
    print(x.shape)
    assert 0
    la_res = np.dot(img1[(x, y)], img2.flatten()).reshape(fft_res.shape)

    print(fft_res)
    print(la_res)
