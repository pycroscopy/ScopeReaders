"""
:class:`~ScopeReaders.generic.image.ImageTranslator` class that extracts the
data from typical image files into :class:`~sidpy.Dataset` objects

Created on Feb 9, 2016

@author: Suhas Somnath, Chris Smith
"""

from __future__ import division, print_function, absolute_import, unicode_literals

import os
import sys
import numpy as np
from PIL import Image
from sidpy.base.num_utils import contains_integers
from sidpy import Dataset, Dimension, Reader

if sys.version_info.major == 3:
    unicode = str
else:
    FileExistsError = ValueError
    FileNotFoundError = ValueError


class ImageReader(Reader):
    """
    Translates data from an image file to sidpy.Dataset
    """

    def __init__(self, *args, **kwargs):
        super(ImageReader, self).__init__(*args, **kwargs)

    @staticmethod
    def _parse_file_path(image_path):
        """
        Returns a list of all files in the directory given by path

        Parameters
        ---------------
        image_path : str
            absolute path to the image file

        Returns
        ----------
        image_path : str
            Absolute file path to the image
        h5_path : str
            absolute path to the desired output HDF5 file.
        """
        if not isinstance(image_path, (str, unicode)):
            raise TypeError("'image_path' argument for ImageTranslator should be a str or unicode")
        if not os.path.exists(os.path.abspath(image_path)):
            raise FileNotFoundError('Specified image does not exist.')
        else:
            image_path = os.path.abspath(image_path)

        return image_path

    def read(self, bin_factor=None, interp_func=Image.BICUBIC,
             normalize=False, **image_args):
        """
        Translates the image in the provided file into a USID HDF5 file

        Parameters
        ----------------
        bin_factor : uint or array-like of uint, optional
            Down-sampling factor for each dimension.  Default is None.
            If specifying different binning for each dimension, please specify as (height binning, width binning)
        interp_func : int, optional. Default = :attr:`PIL.Image.BICUBIC`
            How the image will be interpolated to provide the down-sampled or binned image.
            For more information see instructions for the `resample` argument for :meth:`PIL.Image.resize`
        normalize : boolean, optional. Default = False
            Should the raw image be normalized between the values of 0 and 1
        image_args : dict
            Arguments to be passed to read_image.  Arguments depend on the type of image.

        Returns
        ----------

        """
        image_path = self._parse_file_path(self._input_file_path)

        image = read_image(image_path, **image_args)
        image_parms = dict()
        usize, vsize = image.shape[:2]

        '''
        Check if a bin_factor is given.  Set up binning objects if it is.
        '''
        if bin_factor is not None:
            if isinstance(bin_factor, (list, tuple)):
                if not contains_integers(bin_factor, min_val=1):
                    raise TypeError('bin_factor should contain positive whole integers')
                if len(bin_factor) == 2:
                    bin_factor = tuple(bin_factor)
                else:
                    raise ValueError('Input parameter `bin_factor` must be a length 2 array-like or an integer.\n' +
                                     '{} was given.'.format(bin_factor))

            elif isinstance(bin_factor, int):
                bin_factor = (bin_factor, bin_factor)
            else:
                raise TypeError('bin_factor should either be an integer or an iterable of positive integers')

            if np.min(bin_factor) < 0:
                raise ValueError('bin_factor must consist of positive factors')

            if interp_func not in [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]:
                raise ValueError("'interp_func' argument for ImageTranslator.translate must be one of "
                                 "PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS")

            image_parms.update({'image_binning_size': bin_factor, 'image_PIL_resample_mode': interp_func})
            usize = int(usize / bin_factor[0])
            vsize = int(vsize / bin_factor[1])

            # Unfortunately, we need to make a round-trip through PIL for the interpolation. Not possible with numpy
            img_obj = Image.fromarray(image)
            img_obj = img_obj.resize((vsize, usize), resample=interp_func)
            image = np.asarray(img_obj)

        # Working around occasional "cannot modify read-only array" error
        image = image.copy()

        '''
        Normalize Raw Image
        '''
        if normalize:
            image -= np.min(image)
            image = image / np.float32(np.max(image))

        image_parms.update({'normalized': normalize,
                            'image_min': np.min(image), 'image_max': np.max(image)})

        data_set = Dataset.from_array(image, name='random')

        data_set.data_type = 'image'
        data_set.units = 'a. u.'
        data_set.quantity = 'Intensity'

        data_set.set_dimension(0,
                               Dimension('y', np.arange(usize), units='a. u.',
                                         quantity='Length',
                                         dimension_type='spatial'))
        data_set.set_dimension(1,
                               Dimension('x', np.arange(vsize), units='a. u.',
                                         quantity='Length',
                                         dimension_type='spatial'))

        return data_set

    def can_read(self):
        """
        Tests whether or not the provided file has a .ndata extension
        Returns
        -------

        """
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'csv', 'txt']
        return super(ImageReader, self).can_read(extension=exts)


def read_image(image_path, as_grayscale=True, as_numpy_array=True, *args, **kwargs):
    """
    Read the image file at `image_path` into a numpy array either via numpy (.txt) or via pillow (.jpg, .tif, etc.)

    Parameters
    ----------
    image_path : str
        Path to the image file
    as_grayscale : bool, optional. Default = True
        Whether or not to read the image as a grayscale image
    as_numpy_array : bool, optional. Default = True
        If set to True, the image is read into a numpy array. If not, it is returned as a pillow Image

    Returns
    -------
    image : :class:`numpy.ndarray` or :class:`PIL.Image.Image`
        if `as_numpy_array` is set to True - Array containing the image from the file `image_path`.
        If `as_numpy_array` is set to False - PIL.Image object containing the image within the file - `image_path`.
    """
    ext = os.path.splitext(image_path)[-1]
    if ext in ['.txt', '.csv']:
        if ext == '.csv' and 'delimiter' not in kwargs.keys():
            kwargs['delimiter'] = ','
        img_data = np.loadtxt(image_path, *args, **kwargs)
        if as_numpy_array:
            return img_data
        else:
            img_obj = Image.fromarray(img_data)
            img_obj = img_obj.convert(mode="L")
            return img_obj
    else:
        img_obj = Image.open(image_path)
        if as_grayscale:
            img_obj = img_obj.convert(mode="L", **kwargs)

        if as_numpy_array:
            # Open the image as a numpy array
            return np.asarray(img_obj)
        return img_obj
