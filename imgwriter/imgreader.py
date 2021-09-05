"""
imgreader
~~~~~~~~~

A module for reading image and video files numpy arrays.
"""
from pathlib import Path

import cv2                                            # type: ignore
import numpy as np


def read_image(filepath: str) -> np.ndarray:
    """Read image data from an image file.

    :param filepath: The location of the image file to read.
    :return: A :class:numpy.ndarray object.
    :rtype: numpy.ndarray

    Usage::

        >>> filepath = 'tests/data/__test_save_rgb_image.tiff'
        >>> read_image(filepath)
        array([[[[1.        , 0.49803922, 0.        ],
                 [1.        , 0.49803922, 0.        ],
                 [1.        , 0.49803922, 0.        ]],
        <BLANKLINE>
                [[0.49803922, 0.        , 1.        ],
                 [0.49803922, 0.        , 1.        ],
                 [0.49803922, 0.        , 1.        ]],
        <BLANKLINE>
                [[0.        , 1.        , 0.49803922],
                 [0.        , 1.        , 0.49803922],
                 [0.        , 1.        , 0.49803922]]]])

    Note: The imgwriter package works with both image and video data.
    In an attempt to standardize the output between the two types of
    data, it treats still images as a single frame video. As a result,
    it will add a Z axis to image data from still images.
    """
    # Before wasting time trying to open the file, check if it
    # even exists.
    if not Path(filepath).is_file():
        msg = f'There is no file at {filepath}.'
        raise FileNotFoundError(msg)

    # Read in the data from the image file. Don't change whether it's
    # color or grayscale. If it wasn't readable, puke.
    a = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if a is None:
        msg = f'The file at {filepath} cannot be read.'
        raise ValueError(msg)

    # If the data in the file was unsigned 8-bit integers, convert it
    # to floats in the range 0 <= x <= 1.
    if a.dtype == np.uint8:
        a = a.astype(float)
        a /= 0xff

    # Opencv returns color data from RGB files as BGR. Transform it
    # back to RGB.
    if len(a.shape) == 3:
        a = np.flip(a, -1)

    # Since this module deals with video and still images, it assumes,
    # color images have four dimensions. Opencv reads them in with only
    # three dimensions (two for grayscale). Add a Z axis to the data.
    a = a[np.newaxis, ...]
    return a


def read_video(filepath: str) -> np.ndarray:
    """Capture image data from a video file. Due to the nature of
    video encoding, this doesn't reverse an imgwriter.save(). It
    should be considered experimental for now.
    """
    capture = cv2.VideoCapture(filepath)
    frames = []
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        frames.append(frame)
    capture.release()
    a = np.zeros((len(frames), *frames[0].shape), dtype=frames[0].dtype)
    for i, frame in enumerate(frames):
        a[i] = frame
    return a
