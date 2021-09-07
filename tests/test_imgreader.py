"""
test_imgreader
~~~~~~~~~~~~~~

Unit tests for the imgwriter.imgreader module.
"""
import unittest as ut

import numpy as np

from imgwriter import imgreader as ir


# Test cases.
class ReadImageTestCase(ut.TestCase):
    # Utility methods.
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    # Reused test code.
    def read_grayscale(self, filetype):
        """Given the path to a grayscale image file, return the image's
        data as an array.
        """
        # Expected result.
        exp = np.array([
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
        ], dtype=float)

        # Test data and state.
        filepath = f'tests/data/__test_save_grayscale_image.{filetype}'

        # Run test.
        result = ir.read_image(filepath)

        # Extract actual result.
        # Round the result to two decimal places to allow for the
        # issues with floating point decimals.
        act = np.around(result, 2)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def read_rgb(self, filetype, exp=None):
        """Given the path to a RGB image file, return the image's
        data as an array.
        """
        # Expected result.
        if exp is None:
            exp = np.array([
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
            ], dtype=float)

        # Test data and state.
        filepath = f'tests/data/__test_save_rgb_image.{filetype}'

        # Run test.
        result = ir.read_image(filepath)

        # Extract actual result.
        # Round the result to two decimal places to allow for the
        # issues with floating point decimals.
        act = np.around(result, 2)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    # Test methods.
    def test_read_grayscale_jpg(self):
        """Given the path to a grayscale JPG file, return the image's
        data as an array.
        """
        self.read_grayscale('jpg')

    def test_read_grayscale_png(self):
        """Given the path to a grayscale PNG file, return the image's
        data as an array.
        """
        self.read_grayscale('png')

    def test_read_grayscale_tiff(self):
        """Given the path to a grayscale TIFF file, return the image's
        data as an array.
        """
        self.read_grayscale('tiff')

    def test_read_rgb_jpg(self):
        """Given the path to a RGB JPG file, return the image's
        data as an array.
        """
        # Adjust the expected result to account for how JPEG
        # compression changed the color when we created the test
        # file.
        exp = np.array([
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
        ], dtype=float)
        self.read_rgb('jpg', exp)

    def test_read_rgb_png(self):
        """Given the path to a RGB PNG file, return the image's
        data as an array.
        """
        self.read_rgb('png')

    def test_read_rgb_tiff(self):
        """Given the path to a RGB TIFF file, return the image's
        data as an array.
        """
        self.read_rgb('tiff')

    def test_file_does_not_exist(self):
        """If given the path of a file that doesn't exist, raise a
        FileNotFoundError exception.
        """
        # Test data and state.
        filepath = 'tests/data/spam.jpg'

        # Expected value.
        exp_ex = FileNotFoundError
        exp_msg = f'There is no file at {filepath}.'

        # Run test and determine result.
        with self.assertRaisesRegex(exp_ex, exp_msg):
            ir.read_image(filepath)

    def test_file_not_readable(self):
        """If given the path of a file that isn't a readable image,
        raise a ValueError exception.
        """
        # Test data and state.
        filepath = 'tests/data/__test_not_image.txt'

        # Expected value.
        exp_ex = ValueError
        exp_msg = f'The file at {filepath} cannot be read.'

        # Run test and determine result.
        with self.assertRaisesRegex(exp_ex, exp_msg):
            ir.read_image(filepath)
