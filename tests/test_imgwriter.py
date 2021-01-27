"""
test_imgwriter
~~~~~~~~~~~~~~

Unit tests for the imgwriter.imgwriter module.
"""
import unittest as ut

import numpy as np

from imgwriter import imgwriter as iw


# Test cases.
class CovertColorSpaceTestCase(ut.TestCase):
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    def test_float_grayscale_to_8bit_grayscale(self):
        """Given an array-like object of data in the floating point
        grayscale color space, return a numpy.ndarray object in the
        8-bit grayscale color space.
        """
        # Expected result.
        exp = np.array([
            [
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
            ],
            [
                [0x00, 0x00, 0x00,],
                [0x7f, 0x7f, 0x7f,],
                [0xff, 0xff, 0xff,],
            ],
        ], dtype=np.uint8)

        # Test data and state.
        a = [
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
            [
                [0., 0., 0.,],
                [.5, .5, .5,],
                [1., 1., 1.,],
            ],
        ]
        src_space = ''
        dst_space = 'L'

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_float_grayscale_to_8bit_rgb(self):
        """Given an array-like object of data in the floating point
        grayscale color space, return a numpy.ndarray object in the
        8-bit RGB color space.
        """
        # Expected result.
        exp = np.array([
            [
                [
                    [0x00, 0x00, 0x00,],
                    [0x7f, 0x7f, 0x7f,],
                    [0xff, 0xff, 0xff,],
                ],
                [
                    [0x00, 0x00, 0x00,],
                    [0x7f, 0x7f, 0x7f,],
                    [0xff, 0xff, 0xff,],
                ],
                [
                    [0x00, 0x00, 0x00,],
                    [0x7f, 0x7f, 0x7f,],
                    [0xff, 0xff, 0xff,],
                ],
            ],
            [
                [
                    [0x00, 0x00, 0x00,],
                    [0x00, 0x00, 0x00,],
                    [0x00, 0x00, 0x00,],
                ],
                [
                    [0x7f, 0x7f, 0x7f,],
                    [0x7f, 0x7f, 0x7f,],
                    [0x7f, 0x7f, 0x7f,],
                ],
                [
                    [0xff, 0xff, 0xff,],
                    [0xff, 0xff, 0xff,],
                    [0xff, 0xff, 0xff,],
                ],
            ],
        ], dtype=np.uint8)

        # Test data and state.
        a = [
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
            [
                [0., 0., 0.,],
                [.5, .5, .5,],
                [1., 1., 1.,],
            ],
        ]
        src_space = ''
        dst_space = 'RGB'

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)
