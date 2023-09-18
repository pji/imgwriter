"""
test_imgwriter
~~~~~~~~~~~~~~

Unit tests for the imgwriter.imgwriter module.
"""
from pathlib import Path
import os
import unittest as ut
from unittest.mock import call, MagicMock, patch

import cv2
import numpy as np
import pytest as pt

from imgwriter import imgwriter as iw


# Tests for float_to_unt8.
def test_float_to_uint8_convert():
    """Given an array-like object of floating point values
    between zero and one, :func:`float_to_uint8` should return
    a :class:`numpy.ndarray` object of unsigned 8-bit integers
    between zero and 255.
    """
    a = np.array([[
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],])
    assert (iw._float_to_uint8(a) == np.array([
        [
            [0x00, 0x7f, 0xff,],
            [0x00, 0x7f, 0xff,],
            [0x00, 0x7f, 0xff,],
        ],
    ], dtype=np.uint8)).all()


def test_float_to_uint8_invalid():
    """Given an array-like object of floating point with a value
    greater than one, :func:`float_to_uint8` should raise a
    :class:`ValueError` exception.
    """
    a = np.array([[
        [0., .5, 1.1,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],])
    with pt.raises(ValueError, match='Array values must be 0 >= x >= 1.'):
        _ = iw._float_to_uint8(a)


# Fixtures for save.
@pt.fixture
def eight_bit_rgb(request, tmp_path):
    """Save 8 bit RGB data as an image."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [
        [
            [
                [0xff, 0x7f, 0x00,],
                [0xff, 0x7f, 0x00,],
                [0xff, 0x7f, 0x00,],
            ],
            [
                [0x7f, 0x00, 0xff,],
                [0x7f, 0x00, 0xff,],
                [0x7f, 0x00, 0xff,],
            ],
            [
                [0x00, 0xff, 0x7f,],
                [0x00, 0xff, 0x7f,],
                [0x00, 0xff, 0x7f,],
            ],
        ],
    ]
    path = tmp_path / f'spam.{ext}'
    iw.save(path, a)

    with open(path, 'rb') as fh:
        saved = fh.read()
    yield saved


@pt.fixture
def float_rgb(request, tmp_path):
    """Save float RGB data as an image."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [[
        [
            [1., .5, 0.,],
            [1., .5, 0.,],
            [1., .5, 0.,],
        ],
        [
            [.5, 0., 1.,],
            [.5, 0., 1.,],
            [.5, 0., 1.,],
        ],
        [
            [0., 1., .5,],
            [0., 1., .5,],
            [0., 1., .5,],
        ],
    ],]
    path = tmp_path / f'spam.{ext}'
    iw.save(path, a)

    with open(path, 'rb') as fh:
        saved = fh.read()
    return saved


@pt.fixture
def float_grayscale(request, tmp_path):
    """Save float grayscale data as an image."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [[
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],]
    path = tmp_path / f'spam.{ext}'
    iw.save(path, a)

    with open(path, 'rb') as fh:
        saved = fh.read()
    return saved


@pt.fixture
def float_grayscale_mutlifile(request, tmp_path):
    """Save float grayscale data as an image to multiple files."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [
        [
            [0., .5, 1.,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],
        [
            [0., .5, 1.,],
            [0., .5, 1.,],
            [0., .5, 1.,],
        ],
    ]
    path = tmp_path / f'spam.{ext}'
    iw.save(path, a)

    with open(tmp_path / f'spam_0.{ext}', 'rb') as fh:
        saved0 = fh.read()
    with open(tmp_path / f'spam_1.{ext}', 'rb') as fh:
        saved1 = fh.read()
    return saved0, saved1


@pt.fixture
def uint8_grayscale(request, tmp_path):
    """Save 8 bit grayscale data as an image."""
    marker = request.node.get_closest_marker('ext')
    ext = marker.args[0]
    a = [[
        [0x00, 0x7f, 0xff],
        [0x00, 0x7f, 0xff],
        [0x00, 0x7f, 0xff],
    ],]
    path = tmp_path / f'spam.{ext}'
    iw.save(path, a)

    with open(path, 'rb') as fh:
        saved = fh.read()
    return saved


# Tests for save.
@pt.mark.ext('jpg')
def test_save_8_bit_rgb_as_jpeg(eight_bit_rgb):
    """Given image data in the 8-bit RGB color space and a file
    path ending with "JPG", :func:`save` should save the image
    data to the file path as a JPEG file.
    """
    assert eight_bit_rgb == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03'
        b'\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
        b'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff'
        b'\xc0\x00\x11\x08\x00\x03\x00\x03\x03\x01"\x00\x02\x11\x01\x03'
        b'\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
        b'\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05'
        b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06'
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82'
        b'\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijst'
        b'uvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97'
        b'\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5'
        b'\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3'
        b'\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
        b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01'
        b'\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00'
        b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11'
        b'\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00'
        b'\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91'
        b'\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a'
        b'&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86'
        b'\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4'
        b'\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2'
        b'\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9'
        b'\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7'
        b'\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?'
        b'\x00\xf5\x9f\xf8g\xbf\x87?\xf3\xf9\xe2\xaf\xfc/\xb5\x8f\xfeJ\xa2'
        b'\x8a+\xf3\xbf\xedl\xd7\xfe\x7f\xcf\xff\x00\x02\x97\xf9\x9f\xe4'
        b'\x1f\xfcG\xaf\x1c\xff\x00\xe8\xa9\xcc\xbf\xf0\xbb\x15\xff\x00\xcb'
        b'O\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_8_bit_rgb_as_png(eight_bit_rgb):
    """Given image data in the 8-bit RGB color space and a file
    path ending with "PNG", :func:`save` should save the image
    data to the file path as a PNG file.
    """
    assert eight_bit_rgb == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x02\x00\x00\x00\xd9J"\xe8\x00\x00\x00\x1bIDAT'
        b'\x08\x1dc\xfc_\xcf\x00\x01\x8c\xf5\x0c\xff\x19\xc0\x80\x91'
        b'\xe1\x7f=\x03\x18\x00\x00P\xee\x04~\xdbz\xd5\xe5\x00\x00\x00'
        b'\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tif')
def test_save_8_bit_rgb_as_tiff(eight_bit_rgb):
    """Given image data in the 8-bit RGB color space and a file
    path ending with "TIFF", :func:`save` should save the image
    data to the file path as a TIFF file.
    """
    assert eight_bit_rgb == (
        b'II*\x00\x1a\x00\x00\x00\x80?\xcf\xe0\x08$\x14\x01\x03\x7f\xc1'
        b'\xa0\xb0(P\x02\x02\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00'
        b'\x03\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00'
        b'\x00\x02\x01\x03\x00\x03\x00\x00\x00\xb0\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00\x01'
        b'\x00\x00\x00\x02\x00\x00\x00\x11\x01\x04\x00\x01\x00\x00\x00'
        b'\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00'
        b'\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00\x17\x01'
        b'\x04\x00\x01\x00\x00\x00\x11\x00\x00\x00\x1c\x01\x03\x00\x01'
        b'\x00\x00\x00\x01\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02'
        b'\x00\x00\x00S\x01\x03\x00\x03\x00\x00\x00\xb6\x00\x00\x00\x00'
        b'\x00\x00\x00\x08\x00\x08\x00\x08\x00\x01\x00\x01\x00\x01\x00'
    )


@pt.mark.ext('jpg')
def test_save_float_as_jpeg(float_rgb):
    """Given image data in the floating point RGB color space
    and a file path ending with "JPG", :func:`save` should save
    the image data to the file path as a JPEG file.
    """
    assert float_rgb == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03'
        b'\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
        b'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff'
        b'\xc0\x00\x11\x08\x00\x03\x00\x03\x03\x01"\x00\x02\x11\x01\x03'
        b'\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
        b'\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05'
        b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06'
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82'
        b'\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijst'
        b'uvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97'
        b'\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5'
        b'\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3'
        b'\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
        b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01'
        b'\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00'
        b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11'
        b'\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00'
        b'\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91'
        b'\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a'
        b'&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86'
        b'\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4'
        b'\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2'
        b'\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9'
        b'\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7'
        b'\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?'
        b'\x00\xf5\x9f\xf8g\xbf\x87?\xf3\xf9\xe2\xaf\xfc/\xb5\x8f\xfeJ\xa2'
        b'\x8a+\xf3\xbf\xedl\xd7\xfe\x7f\xcf\xff\x00\x02\x97\xf9\x9f\xe4'
        b'\x1f\xfcG\xaf\x1c\xff\x00\xe8\xa9\xcc\xbf\xf0\xbb\x15\xff\x00\xcb'
        b'O\xff\xd9'
    )


def test_save_float_as_jpeg_not_series(tmp_path):
    """Given image data in the floating point RGB color space
    and a file path ending with "JPG", :func:`save` should save
    the image data to the file path as a JPEG file. If the image
    is not a series, the three dimensional array should still
    be saved as an image.
    """
    a = [
        [
            [1., .5, 0.,],
            [1., .5, 0.,],
            [1., .5, 0.,],
        ],
        [
            [.5, 0., 1.,],
            [.5, 0., 1.,],
            [.5, 0., 1.,],
        ],
        [
            [0., 1., .5,],
            [0., 1., .5,],
            [0., 1., .5,],
        ],
    ]
    path = tmp_path / 'spam_single.jpg'
    iw.save(path, a, as_series=False)

    with open(path, 'rb') as fh:
        saved = fh.read()
    assert saved == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xdb\x00C\x01\x02\x02\x02\x02\x02\x02\x05\x03'
        b'\x03\x05\n\x07\x06\x07\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
        b'\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\xff'
        b'\xc0\x00\x11\x08\x00\x03\x00\x03\x03\x01"\x00\x02\x11\x01\x03'
        b'\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08'
        b'\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05'
        b'\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06'
        b'\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82'
        b'\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijst'
        b'uvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97'
        b'\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5'
        b'\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3'
        b'\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9'
        b'\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01'
        b'\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00'
        b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11'
        b'\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00'
        b'\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91'
        b'\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a'
        b'&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86'
        b'\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4'
        b'\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2'
        b'\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9'
        b'\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7'
        b'\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?'
        b'\x00\xf5\x9f\xf8g\xbf\x87?\xf3\xf9\xe2\xaf\xfc/\xb5\x8f\xfeJ\xa2'
        b'\x8a+\xf3\xbf\xedl\xd7\xfe\x7f\xcf\xff\x00\x02\x97\xf9\x9f\xe4'
        b'\x1f\xfcG\xaf\x1c\xff\x00\xe8\xa9\xcc\xbf\xf0\xbb\x15\xff\x00\xcb'
        b'O\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_float_as_png(float_rgb):
    """Given image data in the floating point space and a file
    path ending with "PNG", :func:`save` should save the image
    data to the file path as a PNG file.
    """
    assert float_rgb == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x02\x00\x00\x00\xd9J"\xe8\x00\x00\x00\x1bIDAT'
        b'\x08\x1dc\xfc_\xcf\x00\x01\x8c\xf5\x0c\xff\x19\xc0\x80\x91'
        b'\xe1\x7f=\x03\x18\x00\x00P\xee\x04~\xdbz\xd5\xe5\x00\x00\x00'
        b'\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tif')
def test_save_float_as_tiff(float_rgb):
    """Given image data in the floating point space and a file
    path ending with "TIFF", :func:`save` should save the image
    data to the file path as a TIFF file.
    """
    assert float_rgb == (
        b'II*\x00\x1a\x00\x00\x00\x80?\xcf\xe0\x08$\x14\x01\x03\x7f\xc1'
        b'\xa0\xb0(P\x02\x02\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00'
        b'\x03\x00\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00'
        b'\x00\x02\x01\x03\x00\x03\x00\x00\x00\xb0\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00\x01'
        b'\x00\x00\x00\x02\x00\x00\x00\x11\x01\x04\x00\x01\x00\x00\x00'
        b'\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00'
        b'\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00\x17\x01'
        b'\x04\x00\x01\x00\x00\x00\x11\x00\x00\x00\x1c\x01\x03\x00\x01'
        b'\x00\x00\x00\x01\x00\x00\x00=\x01\x03\x00\x01\x00\x00\x00\x02'
        b'\x00\x00\x00S\x01\x03\x00\x03\x00\x00\x00\xb6\x00\x00\x00\x00'
        b'\x00\x00\x00\x08\x00\x08\x00\x08\x00\x01\x00\x01\x00\x01\x00'
    )


@pt.mark.ext('jpg')
def test_save_float_grayscale_as_jpg(float_grayscale):
    """Given image data in the float grayscale space and a file
    path ending with "JPG", :func:`save` should save the image
    data to the file path as a JPEG file.
    """
    assert float_grayscale == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_float_grayscale_as_png(float_grayscale):
    """Given image data in the float grayscale space and a file
    path ending with "PNG", :func:`save` should save the image
    data to the file path as a PNG file.
    """
    assert float_grayscale == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tiff')
def test_save_float_grayscale_as_tiff(float_grayscale):
    """Given image data in the float grayscale space and a file
    path ending with "TIFF", :func:`save` should save the image
    data to the file path as a TIFF file.
    """
    assert float_grayscale == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )


@pt.mark.ext('jpg')
def test_save_float_grayscale_as_multiple_jpg(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    JPEG files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )
    assert saved1 == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_float_grayscale_as_multiple_png(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    PNG files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    assert saved1 == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tiff')
def test_save_float_grayscale_as_multiple_tiff(float_grayscale_mutlifile):
    """Given three dimensional image data in the floating point
    grayscale color space and a file path, :func:`save` should
    save the image data as multiple images to the file path as
    TIFF files.
    """
    saved0, saved1 = float_grayscale_mutlifile
    assert saved0 == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )
    assert saved1 == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )


@pt.mark.ext('jpg')
def test_save_uint8_grayscale_as_jpg(uint8_grayscale):
    """Given image data in the 8 bit grayscale space and a file
    path ending with "JPG", :func:`save` should save the image
    data to the file path as a JPEG file.
    """
    assert uint8_grayscale == (
        b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
        b'\x00\x00\xff\xdb\x00C\x00\x02\x01\x01\x01\x01\x01\x02\x01\x01'
        b'\x01\x02\x02\x02\x02\x02\x04\x03\x02\x02\x02\x02\x05\x04\x04'
        b'\x03\x04\x06\x05\x06\x06\x06\x05\x06\x06\x06\x07\t\x08\x06\x07'
        b'\t\x07\x06\x06\x08\x0b\x08\t\n\n\n\n\n\x06\x08\x0b\x0c\x0b\n'
        b'\x0c\t\n\n\n\xff\xc0\x00\x0b\x08\x00\x03\x00\x03\x01\x01\x11'
        b'\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07'
        b'\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04'
        b'\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05'
        b'\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R'
        b'\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHI'
        b'JSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92'
        b'\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8'
        b'\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5'
        b'\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6'
        b'\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfd=\xff'
        b'\x00\x82\x03\xff\x00\xca\x1b?g\xbf\xfb\'\xf0\xff\x00\xe8\xe9k'
        b'\xff\xd9'
    )


@pt.mark.ext('png')
def test_save_uint8_grayscale_as_png(uint8_grayscale):
    """Given image data in the 8 bit grayscale space and a file
    path ending with "PNG", :func:`save` should save the image
    data to the file path as a PNG file.
    """
    assert uint8_grayscale == (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x03\x00\x00'
        b'\x00\x03\x08\x00\x00\x00\x00sC\xeac\x00\x00\x00\x14IDAT\x08'
        b'\x1dcd\xa8o`d\xa8o`d\xa8o\x00\x00\x10\x92\x03\x01\xe0\x97\x89'
        b'\x82\x00\x00\x00\x00IEND\xaeB`\x82'
    )


@pt.mark.ext('tiff')
def test_save_uint8_grayscale_as_tiff(uint8_grayscale):
    """Given image data in the 8 bit grayscale space and a file
    path ending with "PNG", :func:`save` should save the image
    data to the file path as a TIFF file.
    """
    assert uint8_grayscale == (
        b'II*\x00\x12\x00\x00\x00\x80\x00\x0f\xe8\x08\x14\x12\x07'
        b'\x01\x00\x0c\x00\x00\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x01\x01\x03\x00\x01\x00\x00\x00\x03\x00\x00\x00'
        b'\x02\x01\x03\x00\x01\x00\x00\x00\x08\x00\x00\x00\x03\x01'
        b'\x03\x00\x01\x00\x00\x00\x05\x00\x00\x00\x06\x01\x03\x00'
        b'\x01\x00\x00\x00\x01\x00\x00\x00\x11\x01\x04\x00\x01\x00'
        b'\x00\x00\x08\x00\x00\x00\x15\x01\x03\x00\x01\x00\x00\x00'
        b'\x01\x00\x00\x00\x16\x01\x03\x00\x01\x00\x00\x00\x03\x00'
        b'\x00\x00\x17\x01\x04\x00\x01\x00\x00\x00\t\x00\x00\x00'
        b'\x1c\x01\x03\x00\x01\x00\x00\x00\x01\x00\x00\x00=\x01\x03'
        b'\x00\x01\x00\x00\x00\x02\x00\x00\x00S\x01\x03\x00\x01\x00'
        b'\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00'
    )


# Fixtures for save_video.
@pt.fixture
def fpc_video(request, mocker):
    """Test for saving floating point color video."""
    ext = request.node.get_closest_marker('ext').args[0]
    codec = request.node.get_closest_marker('codec').args[0]
    mock_vw = mocker.patch('cv2.VideoWriter')
    a = np.array([
        [
            [
                [1., .5, 0.,],
                [1., .5, 0.,],
                [1., .5, 0.,],
            ],
            [
                [.5, 0., 1.,],
                [.5, 0., 1.,],
                [.5, 0., 1.,],
            ],
            [
                [0., 1., .5,],
                [0., 1., .5,],
                [0., 1., .5,],
            ],
        ],
        [
            [
                [0., 1., .5,],
                [0., 1., .5,],
                [0., 1., .5,],
            ],
            [
                [1., .5, 0.,],
                [1., .5, 0.,],
                [1., .5, 0.,],
            ],
            [
                [.5, 0., 1.,],
                [.5, 0., 1.,],
                [.5, 0., 1.,],
            ],
        ],
        [
            [
                [.5, 0., 1.,],
                [.5, 0., 1.,],
                [.5, 0., 1.,],
            ],
            [
                [0., 1., .5,],
                [0., 1., .5,],
                [0., 1., .5,],
            ],
            [
                [1., .5, 0.,],
                [1., .5, 0.,],
                [1., .5, 0.,],
            ],
        ],
    ])
    path = f'spam.{ext}'
    iw.save_video(path, a, 12, codec)
    return mock_vw.mock_calls


# Tests for save_video.
@pt.mark.codec('mp4v')
@pt.mark.ext('avi')
def test_save_video_fpc_as_avi(fpc_video, mocker):
    """Given image data in the floating point color space,
    :func:`save_video` should save the data as an AVI video
    file.
    """
    actual = fpc_video
    assert actual[0] == mocker.call(
        'spam.avi', cv2.VideoWriter_fourcc(*'mp4v'), 12, (3, 3), True
    )
    assert (actual[1][1][0] == np.array([
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
    ])).all()
    assert (actual[2][1][0] == np.array([
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
    ])).all()
    assert (actual[3][1][0] == np.array([
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
    ])).all()
    assert actual[4] == mocker.call().release()


@pt.mark.codec('mp4v')
@pt.mark.ext('mp4')
def test_save_video_fpc_as_mp4(fpc_video, mocker):
    """Given image data in the floating point color space,
    :func:`save_video` should save the data as an MP4 video
    file.
    """
    actual = fpc_video
    assert actual[0] == mocker.call(
        'spam.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 12, (3, 3), True
    )
    assert (actual[1][1][0] == np.array([
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
    ])).all()
    assert (actual[2][1][0] == np.array([
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
    ])).all()
    assert (actual[3][1][0] == np.array([
        [
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
            [0xff, 0x00, 0x7f],
        ],
        [
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
            [0x7f, 0xff, 0x00],
        ],
        [
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
            [0x00, 0x7f, 0xff],
        ],
    ])).all()
    assert actual[4] == mocker.call().release()


@ut.skip
class SaveVideoTestCase(ut.TestCase):
    # Utility methods.
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    def save_fpc_video(self, ftype, codec='mp4v'):
        """Given image data in the floating point grayscale color
        space, save the data as a video file.
        """
        # Expected result.
        filepath = f'__test_save_fpc_video.{ftype}'
        with open(f'./tests/data/{filepath}', 'rb') as fh:
            exp = fh.read()

        # Test data and state.
        a = [
            [
                [
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                ],
                [
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                ],
                [
                    [0., 1., .5,],
                    [0., 1., .5,],
                    [0., 1., .5,],
                ],
            ],
            [
                [
                    [0., 1., .5,],
                    [0., 1., .5,],
                    [0., 1., .5,],
                ],
                [
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                ],
                [
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                ],
            ],
            [
                [
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                    [.5, 0., 1.,],
                ],
                [
                    [0., 1., .5,],
                    [0., 1., .5,],
                    [0., 1., .5,],
                ],
                [
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                    [1., .5, 0.,],
                ],
            ],
        ]
        framerate = 12

        # Run test.
        try:
            _ = iw.save_video(filepath, a, framerate, codec)

            # Extract actual result.
            with open(filepath, 'rb') as fh:
                act = fh.read()

            # Determine test result.
            self.assertEqual(exp, act)

        # Clean up test.
        finally:
            os.remove(filepath)

    def save_fpg_video(self, ftype, codec='mp4v'):
        """Given image data in the floating point grayscale color
        space, save the data aa a video file.
        """
        # Expected result.
        filepath = f'__test_save_fpg_video.{ftype}'
        with open(f'./tests/data/{filepath}', 'rb') as fh:
            exp = fh.read()

        # Test data and state.
        a = [
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
            [
                [1., 0., .5,],
                [1., 0., .5,],
                [1., 0., .5,],
            ],
            [
                [.5, 1., 0.,],
                [.5, 1., 0.,],
                [.5, 1., 0.,],
            ],
        ]
        framerate = 12

        # Run test.
        try:
            _ = iw.save_video(filepath, a, framerate, codec)

            # Extract actual result.
            with open(filepath, 'rb') as fh:
                act = fh.read()

            # Determine test result.
            self.assertEqual(exp, act)

        # Clean up test.
        finally:
            os.remove(filepath)

    def save_rgb_video(self, ftype, codec='mp4v'):
        """Given image data in the RGB color space, save the data
        as a video file.
        """
        # Expected result.
        filepath = f'__test_save_rgb_video.{ftype}'
        with open(f'./tests/data/{filepath}', 'rb') as fh:
            exp = fh.read()

        # Test data and state.
        a = [
            [
                [
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                ],
                [
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                ],
                [
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                ],
            ],
            [
                [
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                ],
                [
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                ],
                [
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                ],
            ],
            [
                [
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                    [0x7f, 0xff, 0x00],
                ],
                [
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                ],
                [
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                    [0x00, 0x7f, 0xff],
                ],
            ],
        ]
        framerate = 12

        # Run test.
        try:
            _ = iw.save_video(filepath, a, framerate, codec)

            # Extract actual result.
            with open(filepath, 'rb') as fh:
                act = fh.read()

            # Determine test result.
            self.assertEqual(exp, act)

        # Clean up test.
        finally:
            os.remove(filepath)

    # Test methods.
    def test_save_fpg_video_as_avi(self):
        """Given image data in the floating point grayscale color
        space, save the data as an AVI video file.
        """
        self.save_fpg_video('avi', 'MJPG')

    def test_save_fpg_video_as_mp4(self):
        """Given image data in the floating point grayscale color
        space, save the data as an MP4 video file.
        """
        self.save_fpg_video('mp4')

    def test_save_rgb_video_as_avi(self):
        """Given image data in the RGB color space, save the data
        as an AVI video file.
        """
        self.save_rgb_video('avi', 'MJPG')

    def test_save_rgb_video_as_mp4(self):
        """Given image data in the RGB color space, save the data
        as an MP4 video file.
        """
        self.save_rgb_video('mp4')
