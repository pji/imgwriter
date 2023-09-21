.. imgwriter documentation master file, created by
   sphinx-quickstart on Tue Sep 19 07:08:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to imgwriter's documentation!
=====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   self
   /api.rst
   /examples.rst
   /requirements.rst


Why did you write this?
=======================
I'd been working on some code to procedurally generate images and
video. It was getting pretty bloated, and the file IO part seemed
like something that was reasonable to carve off. Because of that, it
is pretty niche. It's really only useful if you are interacting with
images or video as :class:`numpy.ndarray` objects and don't mind some
of the limitations of this package. But, hey, maybe there is someone
else out there who could use it.


How do I run the code?
======================
The best way to get started is to clone the repository to your local
system and take a look at the examples in the example directory.


Is it portable?
===============
Probably. It's written on macOS and uses opencv-python to handle video.
The opencv-python package uses different libraries to handle video on
macOS (QTKit) than it does on Linux/Windows (ffmpeg). I don't think
that affects the specific behavior of imgwriter, but you may run into
problems with missing codecs in other OSes.


Can I install this package from pipenv?
=======================================
Yes, but imgwriter is not currently available through PyPI. You will
need to clone the repository to the system you want to install
imgwriter on and run the following::

    pipenv install path/to/local/copy

Replace `path/to/local/copy` with the path for your local clone of
this repository.


How do I run the tests?
=======================
The unit tests are built using :mod:`pytest`. The `Makefile` has a
shortcut for running these tests::

    make test

The `precommit.py` script contains some extra tests. The shortcut for
running it is::

    make pre


How do I contribute?
====================
At this time, this is code is really just me exploring and learning.
I've made it available in case it helps anyone else, but I'm not really
intending to turn this into anything other than a personal project.

That said, if other people do find it useful and start using it, I'll
reconsider. If you do use it and see something you want changed or
added, go ahead and open an issue. If anyone ever does that, I'll
figure out how to handle it.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
