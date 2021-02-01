"""
test_imgwriter
~~~~~~~~~~~~~~

Unit tests for the imgwriter.imgwriter module.
"""
import os
import unittest as ut
from unittest.mock import call, patch

import numpy as np

from imgwriter import imgwriter as iw


# Test cases.
class FloatToUint8TestCase(ut.TestCase):
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    def test_covert(self):
        """Given an array-like object of floating point values between
        zero and one, return a numpy.ndarray object of unsigned 8-bit
        integers between zero and 255.
        """
        # Expected result.
        exp = np.array([
            [
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
            ],
        ], dtype=np.uint8)
        
        # Test data and state.
        a = [
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
        ]
        
        # Run test.
        act = iw._float_to_uint8(a)
        
        # Determine test result.
        self.assertArrayEqual(exp, act)

    def test_value_out_of_range(self):
        """Given an array-like object of floating point with a value
        greater than one, raise a ValueError exception.
        """
        # Expected result.
        exp_ex = ValueError
        exp_msg = 'Array values must be 0 >= x >= 1.'
        
        exp = np.array([
            [
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
                [0x00, 0x7f, 0xff,],
            ],
        ], dtype=np.uint8)
        
        # Test data and state.
        a = [
            [
                [0., .5, 1.1,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
        ]
        
        # Determine test result.
        with self.assertRaisesRegex(exp_ex, exp_msg):
            
            # Run test.
            act = iw._float_to_uint8(a)


class SaveImageTestCase(ut.TestCase):
    # Utility methods.
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    # Reused test code methods.
    @patch('cv2.imwrite')
    def save_8_bit_rgb(self, exp_path, mock_imwrite):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the given file path.
        """
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
        ]

        # Expected value.
        exp_a = (np.array(a[0], dtype=np.uint8))

        # Run test.
        iw.save_image(exp_path, a)

        # Extract actual result.
        mock_imwrite.call_args
        args = mock_imwrite.call_args.args
        act_path = args[0]
        act_a = args[1]

        # Determine test result.
        self.assertEqual(exp_path, act_path)
        self.assertArrayEqual(exp_a, act_a)
    
    @patch('cv2.imwrite')
    def save_float_rgb(self, exp_path, mock_imwrite):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the given file path.
        """
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
        ]

        # Expected value.
        exp_a = np.array([
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
        ], dtype=np.uint8)
        
        # Run test.
        iw.save_image(exp_path, a)

        # Extract actual result.
        mock_imwrite.call_args
        args = mock_imwrite.call_args.args
        act_path = args[0]
        act_a = args[1]

        # Determine test result.
        self.assertEqual(exp_path, act_path)
        self.assertArrayEqual(exp_a, act_a)

    @patch('cv2.imwrite')
    def save_fpg(self, exp_path, mock_imwrite):
        """Given image data in the floating point grayscale color
        space and a file path, save the image data to the file path.
        """
        # Test data and state.
        a = [
            [
                [0., .5, 1.,],
                [0., .5, 1.,],
                [0., .5, 1.,],
            ],
        ]

        # Expected value.
        exp_a = (np.array(a[0]) * 0xff).astype(np.uint8)

        # Run test.
        iw.save_image(exp_path, a)

        # Extract actual result.
        mock_imwrite.call_args
        args = mock_imwrite.call_args.args
        act_path = args[0]
        act_a = args[1]

        # Determine test result.
        self.assertEqual(exp_path, act_path)
        self.assertArrayEqual(exp_a, act_a)

    @patch('cv2.imwrite')
    def save_fpg_as_multiple_files(self, filetype, mock_imwrite):
        """Given three dimensionalimage data in the floating point
        grayscale color space and a file path, save the image data
        to the file path.
        """
        # Test data and state.
        filepath = f'spam.{filetype}'
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

        # Expected value.
        exp_path_0 = f'spam_0.{filetype}'
        exp_path_1 = f'spam_1.{filetype}'
        exp_a = (np.array(a[0]) * 0xff).astype(np.uint8)

        # Run test.
        iw.save_image(filepath, a)

        # Extract actual result.
        mock_imwrite.call_args
        calls = mock_imwrite.mock_calls
        act_path_0 = calls[0].args[0]
        act_a_0 = calls[0].args[1]
        act_path_1 = calls[1].args[0]
        act_a_1 = calls[1].args[1]

        # Determine test result.
        self.assertEqual(exp_path_0, act_path_0)
        self.assertEqual(exp_path_1, act_path_1)
        self.assertArrayEqual(exp_a, act_a_0)
        self.assertArrayEqual(exp_a, act_a_1)

    @patch('cv2.imwrite')
    def save_l(self, filetype, mock_imwrite):
        """Given image data in the 8-bit grayscale color space and a
        file path, save the image data to the file path.
        """
        # Test data and state.
        a = [
            [
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
                [0x00, 0x7f, 0xff],
            ],
        ]

        # Expected value.
        exp_path = f'spam.{filetype}'
        exp_a = (np.array(a[0], dtype=np.uint8))

        # Run test.
        iw.save_image(exp_path, a)

        # Extract actual result.
        mock_imwrite.call_args
        args = mock_imwrite.call_args.args
        act_path = args[0]
        act_a = args[1]

        # Determine test result.
        self.assertEqual(exp_path, act_path)
        self.assertArrayEqual(exp_a, act_a)

    # Test methods.
    def test_save_8_bit_rgb_as_jpeg(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a JPEG file.
        """
        self.save_8_bit_rgb('spam.jpg')

    def test_save_8_bit_rgb_as_png(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a PNG file.
        """
        self.save_8_bit_rgb('spam.png')

    def test_save_8_bit_rgb_as_tiff(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a TIFF file.
        """
        self.save_8_bit_rgb('spam.tiff')

    def test_save_float_rgb_as_jpg(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a JPG file.
        """
        self.save_float_rgb('spam.jpg')

    def test_save_float_rgb_as_png(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a PNG file.
        """
        self.save_float_rgb('spam.png')

    def test_save_float_rgb_as_tiff(self):
        """Given image data in the 8-bit RGB color space and a file
        path, save the image data to the file path as a TIFF file.
        """
        self.save_float_rgb('spam.tiff')

    def test_save_fpg_as_jpeg(self):
        """Given image data in the floating point grayscale color
        space and a file path, save the image data to the file path
        as a JPEG file.
        """
        self.save_fpg('spam.jpg')

    def test_save_fpg_as_png(self):
        """Given image data in the floating point grayscale color
        space and a file path, save the image data to the file path
        as a PNG file.
        """
        self.save_fpg('spam.png')

    def test_save_fpg_as_tiff(self):
        """Given image data in the floating point grayscale color
        space and a file path, save the image data to the file path
        as a TIFF file.
        """
        self.save_fpg('spam.tiff')

    def test_save_fpg_as_multiple_jpeg(self):
        """Given three dimensionalimage data in the floating point
        grayscale color space and a file path, save the image data
        to the file path as a JPEG file.
        """
        self.save_fpg_as_multiple_files('jpg')

    def test_save_fpg_as_multiple_png(self):
        """Given three dimensionalimage data in the floating point
        grayscale color space and a file path, save the image data
        to the file path as a JPEG file.
        """
        self.save_fpg_as_multiple_files('png')

    def test_save_fpg_as_multiple_tiff(self):
        """Given three dimensionalimage data in the floating point
        grayscale color space and a file path, save the image data
        to the file path as a JPEG file.
        """
        self.save_fpg_as_multiple_files('tiff')

    def test_save_l_as_jpeg(self):
        """Given image data in the 8-bit grayscale color space and a
        file path, save the image data to the file path as a JPEG file.
        """
        self.save_l('jpg')

    def test_save_l_as_png(self):
        """Given image data in the 8-bit grayscale color space and a
        file path, save the image data to the file path as a PNG file.
        """
        self.save_l('png')

    def test_save_l_as_tiff(self):
        """Given image data in the 8-bit grayscale color space and a
        file path, save the image data to the file path as a TIFF file.
        """
        self.save_l('tiff')


class SaveVideoTestCase(ut.TestCase):
    # Utility methods.
    def assertArrayEqual(self, a, b):
        """Given two numpy.ndarray objects, raise an AssertionError if
        they are not equal.
        """
        a_list = a.tolist()
        b_list = b.tolist()
        self.assertListEqual(a_list, b_list)

    def test_save_rgb_video(self):
        """Given image data in the RGB color space, save the data
        as an MP4 video file.
        """
        # Expected result.
        with open('./tests/data/__spam.mp4', 'rb') as fh:
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
        filepath = '__spam.mp4'
        fourcc = 1983148141
        framerate = 12
        framesize = (4, 3)
        iscolor = True
        
        # Expected results.
        exp_a = np.flip(np.array(a, dtype=np.uint8), -1)
        
        # Run test.
        _ = iw.save_video(filepath, a)
        
        # Extract actual result.
        with open(filepath, 'rb') as fh:
            act = fh.read()
        
        # Determine test result.
        self.assertEqual(exp, act)
        
        # Clean up test.
        os.remove(filepath)
            