#! .venv/bin/python
"""
build_data
~~~~~~~~~~

Rebuild the images in the tests/data directory. This uses imgwriter to
create the images, so ensure you know that imgwriter is working as
expected before running this. Otherwise, you risk baking broken
behavior into the tests.
"""
import numpy as np

from imgwriter import save


# Grayscale image files.
filetypes = [
    'jpg',
    'png',
    'tiff',
]
data = [
    [
        [0., .5, 1.,],
        [0., .5, 1.,],
        [0., .5, 1.,],
    ],
]
for filetype in filetypes:
    save(f'tests/data/__test_save_grayscale_image.{filetype}', data)

# Color image files.
filetypes = [
    'jpg',
    'png',
    'tiff',
]
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
for filetype in filetypes:
    save(f'tests/data/__test_save_rgb_image.{filetype}', data)
