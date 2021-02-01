"""
imgwriter
~~~~~~~~~

A Python module for saving arrays as images or video.
"""
from typing import Any

import cv2
import numpy as np


# The ArrayLike type is intended to match numpy's array-like
# descriptive type, which is anything that numpy.array can turn
# into a numpy.ndarray object. That is, ultimately, anything, but
# use of ArrayLike in the type hints is intended to communicate
# the data will be converted to an ndarray, which may have
# consequences.
ArrayLike = Any
X, Y, Z = 2, 1, 0


# Utility functions.
def _float_to_uint8(a: ArrayLike) -> ArrayLike:
    """Convert an array of floating point values to an array of
    unsigned 8-bit integers.
    """
    a = np.array(a)
    if np.max(a) > 1 or np.min(a) < 0:
        msg = 'Array values must be 0 >= x >= 1.'
        raise ValueError(msg)

    a *= 0xff
    return a.astype(np.uint8)


def save_image(filepath: str, a: ArrayLike) -> None:
    a = np.array(a)
    filetype = filepath.split('.')[-1]

    if a.dtype in [float, np.float32]:
        a = _float_to_uint8(a)

    if a.shape[0] == 1:
        a = a[0]
        cv2.imwrite(filepath, a)

    else:
        parts = filepath.split('.')
        filename = '.'.join(parts[:-1])
        for i in range(a.shape[0]):
            framepath = f'{filename}_{i}.{filetype}'
            cv2.imwrite(framepath, a[i])


def save_video(filepath: str, a: ArrayLike, framerate: float = 12) -> None:
    a = np.array(a)
    if a.dtype != np.uint8:
        a = a.astype(np.uint8)
    bgr_a = np.flip(a, -1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    framesize = (a.shape[X], a.shape[Y])
    iscolor = True

    vwriter = cv2.VideoWriter(filepath, fourcc, framerate, framesize, iscolor)
    for i in range(a.shape[Z]):
        vwriter.write(bgr_a[i])
    vwriter.release()
