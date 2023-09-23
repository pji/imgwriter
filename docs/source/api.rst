.. _api:

##########
Public API
##########

The following are the functions that make up the public API
of :mod:`imgwriter`.


Writing Data
============
The following functions will save a :class:`numpy.ndarray` to file:

.. autofunction:: imgwriter.save
.. autofunction:: imgwriter.save_image
.. autofunction:: imgwriter.save_video


Reading Data
============
The following functions will read a file and return a :class:`numpy.ndarray`:

.. autofunction:: imgwriter.read
.. autofunction:: imgwriter.read_image
.. autofunction:: imgwriter.read_video
