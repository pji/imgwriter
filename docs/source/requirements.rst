######################
imgwriter Requirements
######################

The purpose of this document is to detail the requirements for
imgwriter, a Python module to serialize array-like data as an image
or video file. This is an initial take for the purposes of planning.
There may be additional requirements or non-required features added
in the future that are not covered in this document.


*******
Purpose
*******
The purpose of imgwriter is to provide a simple interface for saving
array-like data as an image or video file. It is largely an abstraction
layer for the pillow and opencv-python modules, with a few additions to
make things easier for image and video generation code I write.


***********************
Functional Requirements
***********************
The following are the functional requirements for imgwriter:

*   imgwriter must accept array-like data, such as numpy.ndarray.
*   imgwriter must save two-dimensional array-likes as an image.
*   imgwriter must be able to save three-dimensional array-likes as
    video.
*   imgwriter must be able to save three-dimensional array-likes as
    a series of images.
*   imgwriter must allow users to choose whether to save three-
    dimensional array-likes as video or a series of images.
*   imgwriter must be able to save the image or video to the given
    path.


**********************
Technical Requirements
**********************
The following are the technical requirements for imgwriter:

*   imgwriter must be able to save array-like objects with data in the
    following color spaces:
    *   8-bit RGB
    *   8-bit grayscale
    *   32-bit floating point grayscale
*   imgwriter must be able to save image data in the following formats:
    *   JPEG
    *   TIFF
    *   PNG
*   imgwriter must be able to save video data in the following formats:
    *   MP4
