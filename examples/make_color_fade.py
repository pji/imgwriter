"""
make_color_fade
~~~~~~~~~~~~~~~

Create a video that fades from one color to another.
"""
import argparse
import numpy as np

import imgwriter as iw


R, G, B = 0, 1, 2
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
         start_color: tuple[int, int, int],
         end_color: tuple[int, int, int],
         frames: int,
         framerate: float) -> None:
    """Create a video that fades from one color to another.
    
    :param filepath: The location to save the spacer image.
    :param res: The resolution of the image. This is a tuple of the
        form (x, y), where "x" is the width of the image in pixels and
        "y" is the height of the image in pixels.
    :param start_color: This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the starting color of the fade.
    :param end_color: This is a tuple of integers
        representing the amounts of red, green, and blue (in that
        order) are contained in the final color of the fade.
    :param frames: The number of frames for the transition.
    :param framerate: The frame rate of the video.
    :return: None.
    :rtype: None.
    """
    # Create the array of image data for the fade.
    diff_inc = [-1 * (s - e) / frames for s, e in zip(start_color, end_color)]
    a = np.indices((frames, *res[::-1], 3), dtype=np.float32)[0]
    for c in R, G, B:
        a[:, :, :, c] *= diff_inc[c]
        a[:, :, :, c] += start_color[c]
    a = a.astype(np.uint8)
    
    # Send that image and the save location to imagewriter.save_video.
    iw.save_video(filepath, a, framerate)


if __name__ == '__main__':
    # Define the command line options.
    options = {
        'end_color': {
            'args': ('-e', '--end_color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the last frame in 24-bit hex.',
                'default': '000000'
            }
        },
        'filepath': {
            'args': ('filepath',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'Where to save the spacer image.',
            }
        },
        'framerate': {
            'args': ('-f', '--framerate'),
            'kwargs': {
                'type': int,
                'action': 'store',
                'help': 'The frame rate of the fade.',
                'default': 24,
            }
        },
        'length': {
            'args': ('-l', '--length'),
            'kwargs': {
                'type': int,
                'action': 'store',
                'help': 'The number of frames for the fade.',
                'default': 72,
            }
        },
        'resolution': {
            'args': ('-r', '--resolution',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The resolution of the video.',
                'default': '720p'
            }
        },
        'start_color': {
            'args': ('-s', '--start_color',),
            'kwargs': {
                'type': str,
                'action': 'store',
                'help': 'The color of the first frame in 24-bit hex.',
                'default': 'ffffff'
            }
        },
    }

    # Read the command line arguments.
    p = argparse.ArgumentParser(
        prog='make_color_fade.py',
        description='Create a video of a color fade.',
    )
    for option in options:
        args = options[option]['args']
        kwargs = options[option]['kwargs']
        p.add_argument(*args, **kwargs)
    args = p.parse_args()
    
    # Create the spacer image.
    res = RESOLUTIONS[args.resolution]
    start_color = get_channels(args.start_color)
    end_color = get_channels(args.end_color)
    main(args.filepath, 
         res, 
         start_color, 
         end_color, 
         args.length, 
         args.framerate)
    