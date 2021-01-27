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

    def test_8bit_rgb_to_8bit_bgr(self):
        """Given an array-like object of data in the 8-bit RGB color
        space, return a numpy.ndarray object in the 8-bit BGR color
        space.
        """
        # Expected result.
        exp = np.array([
            [
                [
                    [0x00, 0x7f, 0xff,],
                    [0x00, 0x7f, 0xff,],
                    [0x00, 0x7f, 0xff,],
                ],
                [
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                ],
                [
                    [0x7f, 0xff, 0x00,],
                    [0x7f, 0xff, 0x00,],
                    [0x7f, 0xff, 0x00,],
                ],
            ],
            [
                [
                    [0x00, 0x7f, 0xff,],
                    [0x00, 0x7f, 0xff,],
                    [0x00, 0x7f, 0xff,],
                ],
                [
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                    [0xff, 0x00, 0x7f,],
                ],
                [
                    [0x7f, 0xff, 0x00,],
                    [0x7f, 0xff, 0x00,],
                    [0x7f, 0xff, 0x00,],
                ],
            ],
        ], dtype=np.uint8)

        # Test data and state.
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
        src_space = 'RGB'
        dst_space = 'BGR'

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_8bit_rgb_to_float_grayscale(self):
        """Given an array-like object of data in the 8-bit RGB color
        space, return a numpy.ndarray object in the 8-bit RGB color
        space. This uses a luminosity algorithm where the resulting
        gray value is an uneven mix of the three color channels.
        """
        # Expected result.
        exp = np.array([
            [
                [0.0, 0.498, 1.],
                [0.0, 0.498, 1.],
                [0.0, 0.498, 1.],
            ],
            [
                [0.429, 0.569, 0.,],
                [0.498, 0.498, 0.498],
                [1., 1., 1.],
            ],
        ], dtype=float)

        # Test data and state.
        a = [
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
                    [0x00, 0x7f, 0xff,],
                    [0xff, 0x7f, 0x00,],
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
        ]
        src_space = 'RGB'
        dst_space = ''

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_8bit_bgr_to_float_grayscale(self):
        """Given an array-like object of data in the 8-bit BGR color
        space, return a numpy.ndarray object in the floating point
        grayscale color space. This uses a luminosity algorithm where
        the resulting gray value is an uneven mix of the three color
        channels.
        """
        # Expected result.
        exp = np.array([
            [
                [0.0, 0.498, 1.],
                [0.0, 0.498, 1.],
                [0.0, 0.498, 1.],
            ],
            [
                [0.569, 0.429, 0.,],
                [0.498, 0.498, 0.498],
                [1., 1., 1.],
            ],
        ], dtype=float)

        # Test data and state.
        a = [
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
                    [0x00, 0x7f, 0xff,],
                    [0xff, 0x7f, 0x00,],
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
        ]
        src_space = 'BGR'
        dst_space = ''

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_8bit_grayscale_to_float_grayscale(self):
        """Given an array-like object of data in the 8-bit grayscale
        color space, return a numpy.ndarray in the float grayscale
        space.
        """
        # Expected result.
        exp = np.array([
            [
                [0., .498, 1.,],
                [0., .498, 1.,],
                [0., .498, 1.,],
            ],
            [
                [0., 0., 0.,],
                [.498, .498, .498,],
                [1., 1., 1.,],
            ],
        ], dtype=float)

        # Test data and state.
        a = [
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
        ]
        src_space = 'L'
        dst_space = ''

        # Run test.
        act = iw.convert_color_space(a, src_space, dst_space)

        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_unsupported_source_color_space(self):
        """If an unsupported source color space is given, raise a
        ValueError exception."""
        # Expected values.
        exp_exception = ValueError
        exp_msg = 'SPAM is not a supported color space.'

        # Test data and state.
        a = []
        src_space = 'SPAM'
        dst_space = 'L'

        # Run test and determine result.
        with self.assertRaisesRegex(exp_exception, exp_msg):
            _ = iw.convert_color_space(a, src_space, dst_space)

    def test_unsupported_destination_color_space(self):
        """If an unsupported destination color space is given, raise a
        ValueError exception."""
        # Expected values.
        exp_exception = ValueError
        exp_msg = 'SPAM is not a supported color space.'

        # Test data and state.
        a = []
        src_space = 'L'
        dst_space = 'SPAM'

        # Run test and determine result.
        with self.assertRaisesRegex(exp_exception, exp_msg):
            _ = iw.convert_color_space(a, src_space, dst_space)
