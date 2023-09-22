"""
__init__
~~~~~~~~

The namespace of the :mod:`imgwriter` module.
"""
__all__ = ['imgwriter', 'imgreader']
from imgwriter.common import SUPPORTED
from imgwriter.imgwriter import save, save_image, save_video
from imgwriter.imgreader import read, read_image, read_video
