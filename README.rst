#########
imgwriter
#########

A Python package for saving arrays as images or video.


***********************
Why did you write this?
***********************
I've been working on some code to procedurally generate images and
video. It is getting pretty bloated, and the file IO part seemed
like something that was reasonable to carve off. Because of that, it
is pretty niche. It's really only useful if you are interacting with
images or video as numpy.ndarrays and don't mind some of the
limitations of this package. But, hey, maybe there is someone else
out there who could use it.


**********************
How do I run the code?
**********************
The best way to get started is to clone the repository to your local
system and take a look at the examples in the example directory.


*************************************************
Why is the video reading capability experimental?
*************************************************
It's experimental because reading video is hard. Or, at least creating
a good test for it is awkward. While trying to build unit tests for
imgreader.save_video, I discovered that the mp4v video codec is a lot
more lossy than I expect it to be. It's not just that the colors
change; there isn't the same number of pixels. I'm not exactly sure
what is happening, and until I can figure out the right way to get
the expected number of pixels out of a video file, I'm keeping it as
experimental.


***************
Is it portable?
***************
Probably. It's written on macOS and uses opencv-python to handle video.
The opencv-python package uses different libraries to handle video on
macOS (QTKit) than it does on Linux/Windows (ffmpeg). I don't think
that affects the specific behavior of imgwriter, but you may run into
problems with missing codecs in other OSes.


************************************
Can I install this package from pip?
************************************
Yes, but imgwriter is not currently available through PyPI. You will
need to clone the repository to the system you want to install
statuswriter on and run the following::

    pip install path/to/local/copy

Replace `path/to/local/copy` with the path for your local clone of
this repository.


***********************
How do I run the tests?
***********************
The `precommit.py` script in the root of the repository will run the
unit tests and a few other tests beside. Otherwise, the unit tests
are written with the standard unittest module, so you can run the
tests with::

    python -m unittest discover tests


********************
How do I contribute?
********************
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.
