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
