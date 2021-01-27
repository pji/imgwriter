"""
imgwriter
~~~~~~~~~

A Python module for saving arrays as images or video.
"""
from typing import Any

import numpy as np


# The ArrayLike type is intended to match numpy's array-like
# descriptive type, which is anything that numpy.array can turn
# into a numpy.ndarray object. That is, ultimately, anything, but
# use of ArrayLike in the type hints is intended to communicate
# the data will be converted to an ndarray, which may have
# consequences.
ArrayLike = Any


# Utility functions.
def convert_color_space(a: ArrayLike,
                        src_space: str,
                        dst_space: str) -> np.ndarray:
    """Convert an array from the source color space to the destination
    color space.

    :param a: The array of image data.
    :param src_space: The current color space of the image data.
    :param dst_space: THe colorspace to convert the image data to.
    :return: A :class:numpy.ndarray object.
    :rtype: numpy.ndarray
    """
    supported = ['', 'L', 'RGB', 'BGR']
    if src_space not in supported:
        raise ValueError(f'{src_space} is not a supported color space.')
    if dst_space not in supported:
        raise ValueError(f'{dst_space} is not a supported color space.')

    a = np.array(a)
    channels = {
        1: ['', 'L',],
        3: ['RGB', 'BGR',]
    }
    bitdepth = {
        'float': ['',],
        '8bit': ['L', 'RGB', 'BGR',]
    }

    if src_space in channels[1] and dst_space in channels[3]:
        a = np.tile(a[..., np.newaxis], (1, 1, 1, 3))

    # The luminosity algorithm used here taken from:
    #   https://www.johndcook.com/blog/2009/08/24/algorithms-
    #   convert-color-grayscale/
    elif src_space in channels[3] and dst_space in channels[1]:
        if src_space == 'BGR':
            a = np.flip(a, -1)
            src_space = 'RGB'
        a = a[:, :, :, 0] * .21 + a[:, :, :, 1] * .72 + a[:, :, :, 2] * .07
        a /= 0xff
        a = a.astype(float)
        a = np.around(a, 3)
        src_space = ''

    if src_space in bitdepth['float'] and dst_space in bitdepth['8bit']:
        a *= 0xff
        a = a.astype(np.uint8)
    elif src_space in bitdepth['8bit'] and dst_space in bitdepth['float']:
        a = a.astype(float)
        a /= 0xff
        a = np.around(a, 3)

    if (src_space in ['RGB', 'BGR']
            and dst_space in ['RGB', 'BGR']
            and src_space != dst_space):
        a = np.flip(a, -1)

    return a
