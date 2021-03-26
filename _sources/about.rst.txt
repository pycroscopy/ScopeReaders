SciFiReaders
============

SciFiReaders is a collection of
`sidpy.Reader <https://pycroscopy.github.io/sidpy/_autosummary/sidpy.sid.reader.Reader.html#sidpy.sid.reader.Reader>`_
python classes that extract data and metadata
from scientific data files. The extracted information are returned as
`sidpy.Dataset <https://pycroscopy.github.io/sidpy/_autosummary/sidpy.sid.dataset.Dataset.html#sidpy.sid.dataset.Dataset>`_ objects
which are standardized and exchangable data objects across all packages in the `pycroscopy ecosystem <https://github.com/pycroscopy>`_.

* The information obtained from a ``Reader`` can be:

  * used easily as inputs for various data analysis in the `pycroscopy ecosystem <https://github.com/pycroscopy>`_.
  * written into standardized HDF5 files using functions in
    `pyNSID <https://pycroscopy.github.io/pyNSID/_autosummary/pyNSID.io.hdf_io.write_nsid_dataset.html#pyNSID.io.hdf_io.write_nsid_dataset>`_ and
    `pyUSID <https://pycroscopy.github.io/pyUSID/_autosummary/pyUSID.io.hdf_utils.model.write_sidpy_dataset.html#pyUSID.io.hdf_utils.model.write_sidpy_dataset>`_.
* Please the currently `available list of readers <./available_readers.html>`_. This list will be updated continually.

  * Though the majority of the ``Readers`` in ``SciFiReaders`` are for microscopes, ``SciFiReaders`` is **not** exclusive or limited to any scientific domain.
* The `package is organized <https://github.com/pycroscopy/SciFiReaders/tree/master/SciFiReaders/readers>`_ intuitively by scientific modalities.
* We would be happy to work with you to incorporate your contributions into ``SciFiReaders``!

  * Please see our guide on `how to write your own Reader <./notebooks/00_developing_a_reader/developing_a_reader.html>`_

    * If you already have code that can extract the data and parameters from the data file, it takes only a few lines of code to convert your code to a ``sidpy.Reader`` for ``SciFiReaders``.