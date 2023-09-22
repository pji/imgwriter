#! .venv/bin/python
"""
build_data
~~~~~~~~~~

Rebuild the images in the tests/data directory. This uses imgwriter
to create the images, so ensure you know that imgwriter is working
as expected before running this. Otherwise, you risk baking broken
behavior into the tests.
"""
from subprocess import run

import numpy as np

from imgwriter import SUPPORTED_TYPES, save
from imgwriter.common import VALID_FORMATS, Image, Video
from imgwriter.imgwriter import save_video


# Supported types.
imgtypes = [img for img in VALID_FORMATS if isinstance(img, Image)]
vidtypes = [vid for vid in VALID_FORMATS if isinstance(vid, Video)]

# Grayscale image files.
data = [
    [
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],
]
for img in imgtypes:
    try:
        save(f'tests/data/__test_save_grayscale_image.{img.ext}', data)
    except Exception as ex:
        cls = type(ex)
        raise cls(f'Type: gray {img.ext}. {str(ex)}')

# Color image files.
data = [
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
for img in imgtypes:
    try:
        save(f'tests/data/__test_save_rgb_image.{img.ext}', data)
    except Exception as ex:
        cls = type(ex)
        raise cls(f'Type: {img.ext}. {str(ex)}')

# Grayscale video files.
a = np.zeros((3, 480, 720), dtype=float)
a[1, :, :] = 0.5
a[2, :, :] = 1.0
for vid in vidtypes:
    for codec in vid.codecs:
        try:
            save_video(
                f'tests/data/__test_save_grayscale_video_{codec}.{vid.ext}',
                a, 12, codec
            )
        except Exception as ex:
            cls = type(ex)
            raise cls(f'Type: gray {vid.ext} {codec}. {str(ex)}')

# RGB video files.
a = np.zeros((3, 480, 720, 3), dtype=float)
a[0, :, :, 0] = 1.0
a[1, :, :, 1] = 1.0
a[2, :, :, 2] = 1.0
for vid in vidtypes:
    for codec in vid.codecs:
        try:
            save_video(
                f'tests/data/__test_save_rgb_video_{codec}.{vid.ext}',
                a, 12, codec
            )
        except Exception as ex:
            cls = type(ex)
            raise cls(f'Type: RGB {vid.ext} {codec}. {str(ex)}')

# Color fades.
run([
    'python',
    'examples/make_color_fade.py',
    'tests/data/__test_read_color.mp4',
    '-s', '00ff00',
    '-e', 'ff00ff',
])
run([
    'python',
    'examples/make_color_fade.py',
    'tests/data/__test_make_color_fade_fl.mp4',
    '-s', '00ff00',
    '-e', 'ff00ff',
    '-f', '12',
    '-l', '36',
])
run([
    'python',
    'examples/make_color_fade.py',
    'tests/data/__test_make_color_fade_r.mp4',
    '-s', '00ff00',
    '-e', 'ff00ff',
    '-r', 'dv_ntsc'
])

# Spacers.
run([
    'python',
    'examples/make_spacer.py',
    'tests/data/__test_make_spacer.jpg',
])
run([
    'python',
    'examples/make_spacer.py',
    'tests/data/__test_make_spacer_c.jpg',
    '-c', 'c05632',
])
run([
    'python',
    'examples/make_spacer.py',
    'tests/data/__test_make_spacer_r.jpg',
    '-r', 'dv_ntsc',
])

# Not an image.
with open('tests/data/__test_not_image.txt', 'w') as fh:
    fh.write('')
