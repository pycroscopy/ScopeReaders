# -*- coding: utf-8 -*-

import os
from warnings import warn
import numpy as np

from sidpy.sid import Reader, Dataset, Dimension, DimensionType

try:
    import nanonispy as nap
except ModuleNotFoundError:
    nap = None

# TODO: Adopt any missing features from https://github.com/paruch-group/distortcorrect/blob/master/afm/filereader/nanonisFileReader.py


class NanonisReader(Reader):

    def __init__(self, file_path):
        super().__init__(file_path)
        if nap is None:
            raise ModuleNotFoundError('Please pip install nanonispy to use NanonisReader')
        file_path = os.path.abspath(file_path)
        _, filename = os.path.split(file_path)
        _, self.__file_ext = os.path.splitext(filename)

    @staticmethod
    def _parse_sxm_parms(header_dict, signal_dict):
        """
        Parse sxm files.

        Parameters
        ----------
        header_dict : dict
        signal_dict : dict

        Returns
        -------
        parm_dict : dict

        """
        parm_dict = dict()
        data_dict = dict()

        # Create dictionary with measurement parameters
        meas_parms = {key: value for key, value in header_dict.items()
                      if value is not None}
        info_dict = meas_parms.pop('data_info')
        parm_dict['meas_parms'] = meas_parms

        # Create dictionary with channel parameters
        channel_parms = dict()
        channel_names = info_dict['Name']
        single_channel_parms = {name: dict() for name in channel_names}
        for field_name, field_value, in info_dict.items():
            for channel_name, value in zip(channel_names, field_value):
                single_channel_parms[channel_name][field_name] = value
        for value in single_channel_parms.values():
            if value['Direction'] == 'both':
                value['Direction'] = ['forward', 'backward']
            else:
                direction = [value['Direction']]
        scan_dir = meas_parms['scan_dir']
        for name, parms in single_channel_parms.items():
            for direction in parms['Direction']:
                key = ' '.join((name, direction))
                channel_parms[key] = dict(parms)
                channel_parms[key]['Direction'] = direction
                data = signal_dict[name][direction]
                if scan_dir == 'up':
                    data = np.flip(data, axis=0)
                if direction == 'backward':
                    data = np.flip(data, axis=1)
                data_dict[key] = data
        parm_dict['channel_parms'] = channel_parms

        # Position dimensions
        num_cols, num_rows = header_dict['scan_pixels']
        width, height = header_dict['scan_range']
        pos_names = ['X', 'Y']
        pos_units = ['nm', 'nm']
        pos_vals = np.vstack([
            np.linspace(0, width, num_cols),
            np.linspace(0, height, num_rows),
        ])
        pos_vals *= 1e9
        dims = [Dimension(values, name=name, quantity='Length', units=unit,
                          dimension_type=DimensionType.SPATIAL) for
                name, unit, values
                in zip(pos_names, pos_units, pos_vals)]
        data_dict['Dimensions'] = dims

        return parm_dict, data_dict

    @staticmethod
    def _parse_3ds_parms(header_dict, signal_dict):
        """
        Parse 3ds files.

        Parameters
        ----------
        header_dict : dict
        signal_dict : dict

        Returns
        -------
        parm_dict : dict

        """
        parm_dict = dict()
        data_dict = dict()

        # Create dictionary with measurement parameters
        meas_parms = {key: value for key, value in header_dict.items()
                      if value is not None}
        channels = meas_parms.pop('channels')
        for key, parm_grid in zip(meas_parms.pop('fixed_parameters')
                                  + meas_parms.pop('experimental_parameters'),
                                  signal_dict['params'].T):
            # Collapse the parm_grid along one axis if it's constant
            # along said axis
            if parm_grid.ndim > 1:
                dim_slice = list()
                # Find dimensions that are constant
                for idim in range(parm_grid.ndim):
                    tmp_grid = np.moveaxis(parm_grid.copy(), idim, 0)
                    if np.all(np.equal(tmp_grid[0], tmp_grid[1])):
                        dim_slice.append(0)
                    else:
                        dim_slice.append(slice(None))
                # print(key, dim_slice)
                # print(parm_grid[tuple(dim_slice)])
                parm_grid = parm_grid[tuple(dim_slice)]
            meas_parms[key] = parm_grid
        parm_dict['meas_parms'] = meas_parms

        # Create dictionary with channel parameters and
        # save channel data before renaming keys
        data_channel_parms = dict()
        for chan_name in channels:
            splitted_chan_name = chan_name.split(maxsplit=2)
            if len(splitted_chan_name) == 2:
                direction = 'forward'
            elif len(splitted_chan_name) == 3:
                direction = 'backward'
                splitted_chan_name.pop(1)
            name, unit = splitted_chan_name
            key = ' '.join((name, direction))
            data_channel_parms[key] = {'Name': name,
                                       'Direction': direction,
                                       'Unit': unit.strip('()'),
                                       }
            data_dict[key] = signal_dict.pop(chan_name)
        parm_dict['channel_parms'] = data_channel_parms

        # Add remaining signal_dict elements to data_dict
        data_dict.update(signal_dict)

        # Position dimensions
        nx, ny = header_dict['dim_px']
        if 'X (m)' in parm_dict:
            row_vals = parm_dict.pop('X (m)')
        else:
            row_vals = np.arange(nx, dtype=np.float32)

        if 'Y (m)' in parm_dict:
            col_vals = parm_dict.pop('Y (m)')
        else:
            col_vals = np.arange(ny, dtype=np.float32)
        pos_vals = np.hstack([row_vals.reshape(-1, 1),
                              col_vals.reshape(-1, 1)])
        pos_names = ['X', 'Y']

        dims = [Dimension(values, name=label, quantity='Length', units='nm',
                          dimension_type=DimensionType.SPATIAL)
                for label, values in zip(pos_names, pos_vals.T)]

        # Spectroscopic dimensions
        sweep_signal = header_dict['sweep_signal']
        spec_label, spec_unit = sweep_signal.split(maxsplit=1)
        spec_unit = spec_unit.strip('()')
        # parm_dict['sweep_signal'] = (sweep_name, sweep_unit)
        dc_offset = data_dict['sweep_signal']
        spec_dim = Dimension(dc_offset, quantity='Bias', name=spec_label,
                             units=spec_unit,
                             dimension_type=DimensionType.SPECTRAL)
        dims.append(spec_dim)
        data_dict['Dimensions'] = dims

        return parm_dict, data_dict

    def read(self):
        """

        Returns
        -------

        """
        if self.__file_ext == '.3ds':
            reader = nap.read.Grid
        elif self.__file_ext == '.sxm':
            reader = nap.read.Scan
        elif self.__file_ext == '.dat':
            # reader = nap.read.Spec
            raise NotImplementedError('Cannot read dat files yet')
        else:
            raise ValueError(
                "Nanonis file extension must be one '.3ds', '.sxm', or '.dat'")

        nanonis_data = reader(self._input_file_path)

        header_dict = nanonis_data.header
        signal_dict = nanonis_data.signals

        if self.__file_ext == '.3ds':
            parm_dict, data_dict = self._parse_3ds_parms(header_dict,
                                                         signal_dict)
        elif self.__file_ext == '.sxm':
            parm_dict, data_dict = self._parse_sxm_parms(header_dict,
                                                         signal_dict)
        else:
            parm_dict, data_dict = self._parse_dat_parms(header_dict,
                                                         signal_dict)

        self.parm_dict = parm_dict
        self.data_dict = data_dict

        data_channels = self.parm_dict['channel_parms'].keys()

    def can_read(self):
        """
        Tests whether or not the provided file has a .dm3 extension
        Returns
        -------

        """
        # TODO: Add dat eventually
        if nap is None:
            return False
        return super(NanonisReader, self).can_read(extension=['sxm', '3ds'])

