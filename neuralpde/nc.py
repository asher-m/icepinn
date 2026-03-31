"""
Module containing the objects, methods, and routines to ingest and present
usable sea ice concentration data from NOAA/NSIDC sea ice concentration
datafiles.
"""

import datetime
import netCDF4
import numpy as np

from pathlib import Path
from typing import List, Tuple



def g02202_date_to_date_core(g02202_date) -> datetime.datetime:
    return datetime.date(year=1601, month=1, day=1) + datetime.timedelta(days=int(g02202_date))


g02202_date_to_date = np.vectorize(g02202_date_to_date_core)
"""
Return the date as a datetime object.
"""


class SeaIceV4():
    """
    NOAA/NSIDC version 4 sea ice data class.

    This class includes the methods to ingest NOAA/NSIDC version 4 sea ice
    concentraton files.  See [this link](https://nsidc.org/data/g02202/versions/4)
    for more information.

    Attributes:
        seaice_conc:                 Array of fractional sea ice concentration values like (time x ygrid x xgrid)  Values range [0., 1.].
        seaice_stdev:                Array of sea ice concentration stdev values like (time x ygrid x xgrid).  Values range [0., 1.].

        flag_missing:                Boolean array of missing data flags like (time x ygrid x xgrid).
        flag_land:                   Boolean array of land (land not adjacent to ocean) like (time x ygrid x xgrid).
        flag_coast:                  Boolean array of coast (land adjacent to ocean) flags like (time x ygrid x xgrid).
        flag_lake:                   Boolean array of lake data flags like (time x ygrid x xgrid).
        flag_hole:                   Boolean array of imaging hole flags like (time x ygrid x xgrid).

        latitude:                    Array of latitude coordinates as degrees north like (ygrid x xgrid).
        longitude:                   Array of longitude coordinates as degrees east like (ygrid x xgrid).
        meters_x:                    Array of x-offsets in meters of the center of each cell from the projection center like (xgrid).
        meters_y:                    Array of y-offsets in meters of the center of each cell from the projection center like (ygrid).
    """

    def __init__(self, nc_files: List[str] | List[Path]) -> None:
        """
        Initialize sea ice data in NOAA/NSIDC version 4 format.

        Args:
            nc_files (list of str or list of pathlib.Path):     .nc file to be opened
        """
        if len(nc_files) < 1: raise ValueError('Received an empty list of files!')

        self._nc_files = nc_files

        date = []
        seaice_conc, seaice_stdev = [], []
        flag_missing, flag_land, flag_coast, flag_lake, flag_hole = [], [], [], [], []
        latitude, longitude = [], []
        meters_x, meters_y = [], []
        for f in self._nc_files:
            with netCDF4.Dataset(f) as d:
                # handle date
                date.append(g02202_date_to_date(np.array(d.variables['time']).astype(int)))

                # handle sea ice concentration
                s = np.array(d['cdr_seaice_conc'])
                flag_missing.append(s == 255)  # get flags
                flag_land.append(s == 254)
                flag_coast.append(s == 253)
                flag_lake.append(s == 252)
                flag_hole.append(s == 251)
                s[s >= 251] = np.nan  # mask out flags
                seaice_conc.append(s)

                # handle stdev
                s = np.array(d['stdev_of_cdr_seaice_conc'])
                s[s == -1] = np.nan  # mask out missing data
                seaice_stdev.append(s)

                # handle everything else
                latitude.append(np.array(d['latitude']))
                longitude.append(np.array(d['longitude']))
                meters_x.append(np.array(d['xgrid']))
                meters_y.append(np.array(d['ygrid']))

        # check if all the grids match up
        for i in range(1, len(latitude)):
            if not np.all(np.isclose(latitude[i], latitude[0])) or \
               not np.all(np.isclose(longitude[i], longitude[0])) or \
               not np.all(np.isclose(meters_x[i], meters_x[0])) or \
               not np.all(np.isclose(meters_y[i], meters_y[0])):
                raise ValueError('Grid changed between files!  Cannot proceed!')

        # assign attributes
        self.date = np.concatenate(date)

        self.seaice_conc = np.concatenate(seaice_conc).transpose((0, 2, 1))
        self.seaice_stdev = np.concatenate(seaice_stdev).transpose((0, 2, 1))

        self.flag_missing = np.concatenate(flag_missing).transpose((0, 2, 1))
        self.flag_land = np.concatenate(flag_land).transpose((0, 2, 1))
        self.flag_coast = np.concatenate(flag_coast).transpose((0, 2, 1))
        self.flag_lake = np.concatenate(flag_lake).transpose((0, 2, 1))
        self.flag_hole = np.concatenate(flag_hole).transpose((0, 2, 1))

        self.latitude = latitude[0].T
        self.longitude = longitude[0].T
        self.meters_x = meters_x[0]
        self.meters_y = meters_y[0]


class SeaIceV5():
    """
    NOAA/NSIDC version 5 sea ice data class.

    This class includes the methods to ingest NOAA/NSIDC version 5 sea ice
    concentraton files.  See [this link](https://nsidc.org/data/g02202/versions/5)
    for more information.
    """
    def __init__(self):
        raise NotImplementedError()


def check_boundaries(indices: List[int] | Tuple[int] | np.ndarray[int], d: SeaIceV4 | SeaIceV5) -> None:
    """
    Verify that boundaries at each index in `indices` are the same (constant,) and fails if not.

    Args:
        indices:        List or tuple of indices to check, (you probably want these to be adjacent.)
        d:              Sea ice data object.
    """ 
    indices = list(indices)   
    if not np.all(d.flag_missing[indices[0]] == d.flag_missing[indices]) or \
        not np.all(d.flag_land[indices[0]] == d.flag_land[indices]) or \
        not np.all(d.flag_coast[indices[0]] == d.flag_coast[indices]) or \
        not np.all(d.flag_lake[indices[0]] == d.flag_lake[indices]) or \
        not np.all(d.flag_hole[indices[0]] == d.flag_hole[indices]):
        raise ValueError('Found inconsistent boundaries in data!')
