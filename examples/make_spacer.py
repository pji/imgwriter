"""
make_spacer
~~~~~~~~~~~

Create a black JPG that can be used as a spacer in a video.
"""
import argparse
from textwrap import dedent

import numpy as np

import imgwriter as iw


RESOLUTIONS = {
    'dv_ntsc': (720, 480),
    'd1_ntsc': (720, 486),
    'dv_pal': (720, 576),
    'd1_pal': (720, 576),
    'dvcpro_hd_720p': (960, 720),
    'dvcpro_hd_1080_59i': (1280, 1080),
    'dvcpro_hd_1080_50i': (1440, 1080),
    'hdv_1080i': (1440, 1080),
    'hdv_1080p': (1440, 1080),
    'sony_hdcam': (1440, 1080),
    'sony_hdcam_sr': (1440, 1080),
    'academy_2x': (1828, 1332),
    'full_aperature_native_2x': (2048, 1556),
    'academy_4x': (3656, 2664),
    'full_aperature_4x': (4096, 3112),
    '720p': (1280, 720),
    '1080p': (1920, 1080),
    '4k': (3840, 2160),
    '8k': (7680, 4320),
    '16k': (15360, 8640),
}


def get_channels(color: str) -> tuple[int, int, int]:
    """Convert a 24-bit hex color string into an RGB color.

    :param color: A 24-bit hexadecimal color string, like is commonly
        used in HTML and CSS.
    :return: A :class:tuple object.
    :rtype: tuple
    """
    r = int(color[:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:], 16)
    return (r, g, b)


def main(filepath: str,
         res: tuple[int, int],
         color: tuple[int, int, int]) -> None:
    """Create a color image that can be used as a spacer in a video.

    :param filepath: The location to save the spacer image.
    :param res: The resolution of the image. This is a tuple of the
        form (x, y), where "x" is the width of the image in pixels and
        "y" is the height of the image in pixels.
    :param color: The color of the image. This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the color.
    :return: None.
    :rtype: None.
    """
    # Create the array of image data for the spacer image.
    a = np.zeros((1, *res[::-1], 3), dtype=int)
    for c in 0, 1, 2:
        a[:, :, :, c] = color[c]

    # Send that image and the save location to imagewriter.save_image.
    iw.save(filepath, a)


if __name__ == '__main__':
    # Define the command line options.
    resolutions = tuple(key for key in RESOLUTIONS)
    resolution_descr = '\n'.join(
        f'  * {key} ({RESOLUTIONS[key][0]}\u00d7{RESOLUTIONS[key][1]})'
        for key in RESOLUTIONS
    )
    options = {
        'filepath': {
            'args': ('filepath',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'Where to save the spacer image.',
            }
        },
        'resolution': {
            'args': ('-r', '--resolution',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'choices': resolutions,
                'help': 'The resolution of the video. See options below.',
                'metavar': 'RESOLUTION',
                'default': '720p'
            }
        },
        'color': {
            'args': ('-c', '--color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the frame in 24-bit hex.',
                'default': '000000'
            }
        },
    }

    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='make_spacer.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Create a spacer image for video.',
        epilog=dedent('''\
        RESOLUTIONS
        -----------
        The following resolutions are available options:

        ''') + resolution_descr
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    args = p.parse_args()

    # Create the spacer image.
    res = RESOLUTIONS[args.resolution]
    color = get_channels(args.color)
    main(args.filepath, res, color)
