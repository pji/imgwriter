"""
imgwriter
~~~~~~~~~

A Python module for saving arrays as images or video.
"""
import logging
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Union

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray

from imgwriter.common import SUPPORTED, Image, UnsupportedFileType, Video


# Importable names.
__all__ = [
    "save", "save_image", "save_video",
    "write", "write_image", "write_video",
]


# Constants to replace indices with axis names for readability.
X, Y, Z = 2, 1, 0


# Types.
Saver = Union[
    Callable[[Union[str, Path], NDArray[np.uint8], bool], None],
    Callable[[Union[str, Path], NDArray[np.uint8], float, str], None]
]
WrappedSaver = Union[
    Callable[[Union[str, Path], ArrayLike, bool], None],
    Callable[[Union[str, Path], ArrayLike, float, str], None]
]


# Decorators
def uses_opencv(fn: Saver) -> WrappedSaver:
    """Condition the image data for use by opencv prior to saving."""
    @wraps(fn)
    def wrapper(
        filepath: Union[str, Path], a: ArrayLike, *args, **kwargs
    ) -> None:
        # Convert the image data to an array just in case we were passed
        # something else.
        a = np.array(deepcopy(a))

        # While TIFFs can handle 32-bit floats, JPGs and PNGs can't, so
        # rather than having TIFFs as an exception, just convert all floats
        # to unsigned 8-bit integers.
        if a.dtype in [float, np.float32]:
            a = _float_to_uint8(a)

        # If the data isn't a float but not a unsigned 8-bit integer,
        # we assume it's in the right scale. So, just convert to a
        # unsigned 8-bit integer.
        elif a.dtype != np.uint8:
            a = a.astype(np.uint8)

        # opencv saves color data in BGR order, so RGB data needs to be
        # flipped to BGR.
        if len(a.shape) == 4:
            a = np.flip(a, -1)
        elif (len(a.shape) == 3
                and 'as_series' in kwargs
                and not kwargs['as_series']):
            a = np.flip(a, -1)

        return fn(filepath, a, *args, **kwargs)
    return wrapper


# Utility functions.
def _float_to_uint8(a: ArrayLike) -> NDArray[np.uint8]:
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


def write(filepath: Union[str, Path], a: ArrayLike, *args, **kwargs) -> None:
    """Save an array of image data to file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the format used
        by the file.
    :param a: The array of image data.
    :return: None.
    :rtype: None.

    Supported File Types
    --------------------
    :mod:`imgwriter` uses :mod:`cv2` for all save operations. The
    formats supported by :mod:`cv2` varies depending on operating
    system and software installed. Since I need to check the file
    type to determine whether an image or a video is being saved,
    I have to limit the supported file formats to ones I think
    are supported across platforms.
    """
    filepath = Path(filepath)
    ftype = filepath.suffix.casefold()[1:]
    save_as = SUPPORTED[ftype]
    if isinstance(save_as, Image):
        save_fn = save_image
    elif isinstance(save_as, Video):
        save_fn = save_video
    else:
        raise UnsupportedFileType(f'{ftype}')
    save_fn(filepath, a, *args, **kwargs)


@uses_opencv
def write_image(
    filepath: Union[str, Path],
    a: NDArray[np.uint8],
    as_series: bool = True
) -> None:
    """Save an array of image data as an image file.

    :param filepath: The location and name of the file that will
        be saved. The file extension will determine the format used
        by the file. The data needs to be either in an RGB or grayscale
        color space.
    :param a: The array of image data.
    :param as_series: (Optional.) Whether the array is intended to be a
        series of images.
    :return: None.
    :rtype: None.
    """
    filepath = Path(filepath)

    # If the array isn't a series of images, just save what is given.
    if not as_series:
        cv2.imwrite(str(filepath), a)

    # If there is just 1 item in the Z axis, save the image data as
    # a single image.
    elif a.shape[Z] == 1:
        a = a[Z]
        cv2.imwrite(str(filepath), a)

    # If there are multiple items in the Z axis, save the image data
    # as multiple images.
    else:
        fileparent = filepath.parent
        filename = filepath.stem
        filetype = filepath.suffix
        for i in range(a.shape[Z]):
            framepath = str(fileparent / f'{filename}_{i}{filetype}')
            cv2.imwrite(framepath, a[i])


@uses_opencv
def write_video(
    filepath: Union[str, Path],
    a: NDArray[np.uint8],
    framerate: float = 12.0,
    codec: str = 'mp4v'
) -> None:
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
    # cv2.VideoWriter requires a string rather than a Path.
    filepath = str(filepath)

    fourcc = cv2.VideoWriter_fourcc(*codec)
    framesize = (a.shape[X], a.shape[Y])
    iscolor = False
    if len(a.shape) == 4:
        iscolor = True

    vwriter = cv2.VideoWriter(
        filepath, fourcc, framerate, framesize, iscolor
    )
    for i in range(a.shape[Z]):
        vwriter.write(a[i])
    vwriter.release()


# Function aliases.
save = write
save_image = write_image
save_video = write_video
