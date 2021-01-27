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
    """
    a = np.array(a)
    channels = {
        1: ['', 'L',],
        3: ['RGB',]
    }
    bitdepth = {
        'float': ['',],
        '8bit': ['L', 'RGB',]
    }
    
    if src_space in channels[1] and dst_space in channels[3]:
        a = np.tile(a[..., np.newaxis], (1, 1, 1, 3))
    
    if src_space in bitdepth['float'] and dst_space in bitdepth['8bit']:
        a *= 0xff
        a = a.astype(np.uint8)
    
    return a
