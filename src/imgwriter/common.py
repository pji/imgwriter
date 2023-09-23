"""
common
~~~~~~

Configuration values used by :mod:`imgwriter`.
"""
from collections import defaultdict
from dataclasses import dataclass
from typing import Union


# Exceptions.
class UnsupportedFileType(TypeError):
    """The given file type isn't supported."""


# Dataclasses.
@dataclass
class Image:
    ext: str
    description: str = ''


@dataclass
class Video:
    ext: str
    description: str = ''
    codecs: tuple[str, ...] = tuple()


# Common data.
RESOLUTIONS: dict[str, tuple[int, int]] = {
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
VALID_FORMATS: list[Union[Image, Video]] = [
    Image('bmp', 'Windows bitmap'),
    Image('dib', 'Windows bitmap'),

    Image('hdr', 'Radiance HDR'),
    Image('pic', 'Radiance HDR'),

    Image('jpe', 'JPEG'),
    Image('jpg', 'JPEG'),
    Image('jpeg', 'JPEG'),

    Image('png', 'portable network graphics'),

    Image('pnm', 'portable image format'),

    Image('ras', 'Sun raster'),
    Image('sr', 'Sun raster'),

    Image('tif', 'TIFF'),
    Image('tiff', 'TIFF'),

    Image('webp', 'WebP'),

    Video('avi', 'Audio Video Interleave', ('avc1', 'mp4v',)),
    Video('mov', 'QuickTime movie', ('avc1', 'hev1', 'mp4v',)),
    Video('mp4', 'MPEG-4 part 14', ('avc1', 'hev1', 'mp4v',)),
]

# Register supported types. This is also used to determine whether the
# user is trying to save data as a still image or video.
SUPPORTED: defaultdict[str, Union[Image, Video, None]] = defaultdict(
    None, {format.ext: format for format in VALID_FORMATS}
)
