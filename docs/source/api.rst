.. _api:

##########
Public API
##########

The following are the functions that make up the public API
of :mod:`imgwriter`.


Writing Data
============
The following functions will save a :class:`numpy.ndarray` to an image
or video file:

.. autofunction:: imgwriter.write
.. autofunction:: imgwriter.write_image
.. autofunction:: imgwriter.write_video


Aliases
-------
The following functions are aliases to the functions given above. They
exist for backwards compatibility, but may go away in the future:

.. autofunction:: imgwriter.save
.. autofunction:: imgwriter.save_image
.. autofunction:: imgwriter.save_video


Reading Data
============
The following functions will read a file and return a :class:`numpy.ndarray`:

.. autofunction:: imgwriter.read
.. autofunction:: imgwriter.read_image
.. autofunction:: imgwriter.read_video


Aliases
-------
The following functions are aliases to the functions given above. They
exist to be more predictable names if you are used to using the `save`
functions from :mod:`imgwriter.imgwriter`, but they may go away in the
future:

.. autofunction:: imgwriter.load
.. autofunction:: imgwriter.load_image
.. autofunction:: imgwriter.load_video

