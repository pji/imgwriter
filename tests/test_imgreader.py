"""
test_imgreader
~~~~~~~~~~~~~~

Unit tests for the imgwriter.imgreader module.
"""
import numpy as np
import pytest as pt

from imgwriter import imgreader as ir


# Fixtures.
@pt.fixture
def image(request):
    """A common test for :func:`read_image`."""
    path = request.node.get_closest_marker('path').args[0]
    path = f'tests/data/{path}'
    a = ir.read_image(path)
    return np.around(a, 2)


@pt.fixture
def image_as_vid(request):
    """A common test for :func:`read_image` with `as_video`."""
    path = request.node.get_closest_marker('path').args[0]
    path = f'tests/data/{path}'
    a = ir.read_image(path, as_video=True)
    return np.around(a, 2)


# Tests for read_image.
@pt.mark.path('__test_save_grayscale_image.jpg')
def test_read_image_grayscale_jpg(image):
    """Given the path to a grayscale JPG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ])).all()


@pt.mark.path('__test_save_grayscale_image.png')
def test_read_image_grayscale_png(image):
    """Given the path to a grayscale PNG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ])).all()


@pt.mark.path('__test_save_grayscale_image.tiff')
def test_read_image_grayscale_tiff(image):
    """Given the path to a grayscale TIFF file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ])).all()


@pt.mark.path('__test_save_grayscale_image.jpg')
def test_read_image_grayscale_jpg_as_vid(image_as_vid):
    """Given the path to a grayscale JPG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image_as_vid == np.array([[
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ]])).all()


@pt.mark.path('__test_save_rgb_image.jpg')
def test_read_image_rgb_jpg(image):
    """Given the path to a RGB JPG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
        [
            [
                [.92, .41, .66,],
                [.92, .41, .66,],
                [.92, .41, .66,],
            ],
            [
                [.59, .08, .33,],
                [.59, .08, .33,],
                [.59, .08, .33,],
            ],
            [
                [0., 1., .51,],
                [0., 1., .51,],
                [0., 1., .51,],
            ],
        ],
    ])).all()


@pt.mark.path('__test_save_rgb_image.png')
def test_read_image_rgb_png(image):
    """Given the path to a RGB PNG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
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
    ])).all()


@pt.mark.path('__test_save_rgb_image.tiff')
def test_read_image_rgb_tiff(image):
    """Given the path to a TIFF PNG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image == np.array([
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
    ])).all()


@pt.mark.path('__test_save_rgb_image.jpg')
def test_read_image_rgb_jpg_as_vid(image_as_vid):
    """Given the path to a RGB JPG file, :func:`read_image`
    return the image's data as an :class:`numpy.ndarray`.
    """
    assert (image_as_vid == np.array([[
        [
            [
                [.92, .41, .66,],
                [.92, .41, .66,],
                [.92, .41, .66,],
            ],
            [
                [.59, .08, .33,],
                [.59, .08, .33,],
                [.59, .08, .33,],
            ],
            [
                [0., 1., .51,],
                [0., 1., .51,],
                [0., 1., .51,],
            ],
        ],
    ]])).all()


def test_read_image_file_does_not_exist():
    """If given the path of a file that doesn't exist, :func:`save_image`
    should raise a :class:`FileNotFoundError`.
    """
    path = 'tests/data/spam.jpg'
    with pt.raises(
        FileNotFoundError,
        match=f'There is no file at {path}.'
    ):
        _ = ir.read_image(path)


def test_read_image_file_not_readablet():
    """If given the path of a file that isn't a readable image,
    :func:`read_image` should raise a :class:`ValueError` exception.
    """
    path = 'tests/data/__test_not_image.txt'
    with pt.raises(
        ValueError,
        match=f'The file at {path} cannot be read.'
    ):
        _ = ir.read_image(path)
