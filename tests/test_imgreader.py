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
    a = ir.read(path)
    return np.around(a, 2)


@pt.fixture
def image_as_vid(request):
    """A common test for :func:`read_image` with `as_video`."""
    path = request.node.get_closest_marker('path').args[0]
    path = f'tests/data/{path}'
    a = ir.read_image(path, as_video=True)
    return np.around(a, 2)


@pt.fixture
def video_data():
    """An array of video data for testing."""
    frames = 72
    res = [1280, 720]
    start_color = (0x00, 0xff, 0x00)
    end_color = (0xff, 0x00, 0xff)
    diff_inc = [-1 * (s - e) / frames for s, e in zip(start_color, end_color)]
    a = np.indices((frames, *res[::-1], 3), dtype=np.float32)[0]
    for c in 0, 1, 2:
        a[:, :, :, c] *= diff_inc[c]
        a[:, :, :, c] += start_color[c]
    yield a.astype(np.uint8)


# Tests for load.
def test_load_is_alias_for_read():
    """:func:`read` is an alias for :func:`save`."""
    assert ir.load is ir.read
    assert ir.load_image is ir.read_image
    assert ir.load_video is ir.read_video


# Tests for read.
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


def test_read_color_mp4(video_data):
    """Given a path to an MP4 file, :func:`read_video` should return the
    contents of the video as a :class:`numpy.ndarray`.
    """
    path = 'tests/data/__test_read_color.mp4'
    a = ir.read(path)

    # The compression makes it hard to predict the exact color values
    # of each pixel in the output. The following checks to see if there
    # is any variance greater than ~2%. The data type has to change to
    # `int` because `numpy.uint8` is unsigned, so any negative values
    # roll over.
    assert (np.abs(a.astype(int) - video_data.astype(int)) <= 5).all()


# Tests for read_image.
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


# Tests for read_video.
def test_read_video_color_mp4(video_data):
    """Given a path to an MP4 file, :func:`read_video` should return the
    contents of the video as a :class:`numpy.ndarray`.
    """
    path = 'tests/data/__test_read_color.mp4'
    a = ir.read_video(path)

    # The compression makes it hard to predict the exact color values
    # of each pixel in the output. The following checks to see if there
    # is any variance greater than ~2%. The data type has to change to
    # `int` because `numpy.uint8` is unsigned, so any negative values
    # roll over.
    assert (np.abs(a.astype(int) - video_data.astype(int)) <= 5).all()
