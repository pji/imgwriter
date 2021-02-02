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
def _float_to_uint8(a: ArrayLike) -> np.ndarray:
    """Convert an array of floating point values to an array of
    unsigned 8-bit integers.

    :param a: The array of image data to convert to unsigned 8-bit
        integers.
    :return: A :class:numpy.ndarray object.
    :rtype: numpy.ndarray
    """
    a = np.array(a)
    if np.max(a) > 1 or np.min(a) < 0:
        msg = 'Array values must be 0 >= x >= 1.'
        raise ValueError(msg)

    a *= 0xff
    return a.astype(np.uint8)


def save_image(filepath: str, a: ArrayLike) -> None:
    """Save an array of image data as an image file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the format used
        by the file. The data needs to be either in an RGB or grayscale
        color space.
    :param a: The array of image data.
    :return: None.
    :rtype: None.
    """
    # Convert the image data to an array just in case we were passed
    # something else.
    a = np.array(a)

    # While TIFFs can handle 32-bit floats, JPGs and PNGs can't, so
    # rather than having TIFFs as an exception, just convert all floats
    # to unsigned 8-bit integers.
    if a.dtype in [float, np.float32]:
        a = _float_to_uint8(a)

    # opencv saves color data in BGR order, so RGB data needs to be
    # flipped to BGR.
    if len(a.shape) == 4:
        a = np.flip(a, -1)

    # If there is just 1 item in the Z axis, save the image data as
    # a single image.
    if a.shape[Z] == 1:
        a = a[Z]
        cv2.imwrite(filepath, a)

    # If there are multiple items in the Z axis, save the image data
    # as multiple images.
    else:
        parts = filepath.split('.')
        filetype = parts[-1]
        filename = '.'.join(parts[:-1])
        for i in range(a.shape[Z]):
            framepath = f'{filename}_{i}.{filetype}'
            cv2.imwrite(framepath, a[i])


def save_video(filepath: str,
               a: ArrayLike,
               framerate: float = 12,
               codec: str = 'mp4v') -> None:
    """Save an array of image data as a video file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the container
        type used for the file.
    :param a: The array of image data.
    :param framerate: (Optional.) The number of frames the video will
        play per second.
    :param codec: (Optional.) The codec used to encode the image data
        into video. The exact list of supported codecs depends upon
        the operating system. Per the opencv documentation, Linux and
        Windows will tend to use the list supported by ffmpeg and
        macOS will use the list suported by QTKit.
    :return: None.
    :rtype: None.
    """
    a = np.array(a)
    if a.dtype in [float, np.float32]:
        a = _float_to_uint8(a)
    elif a.dtype != np.uint8:
        a = a.astype(np.uint8)
    filetype = filepath.split('.')[-1]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    framesize = (a.shape[X], a.shape[Y])
    iscolor = False
    if len(a.shape) == 4:
        a = np.flip(a, -1)
        iscolor = True

    vwriter = cv2.VideoWriter(filepath, fourcc, framerate, framesize, iscolor)
    for i in range(a.shape[Z]):
        vwriter.write(a[i])
    vwriter.release()
