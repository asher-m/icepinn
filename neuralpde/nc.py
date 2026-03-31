"""
Module containing the objects, methods, and routines to ingest and present
usable sea ice concentration data from NOAA/NSIDC sea ice concentration
datafiles.
"""

import datetime
import netCDF4
import numpy as np
import numpy.typing as npt

from pathlib import Path
from typing import List, Tuple



class SeaIceV6():
    """
    NOAA/NSIDC version 6 sea ice data class.

    This class includes the methods to ingest NOAA/NSIDC version 6 sea ice
    concentraton files.  See [this link](https://nsidc.org/data/g02202/versions/6)
    for more information.

    Attributes:
        date:                        Array of dates.
    
        seaice_conc:                 Array of fractional sea ice concentration values like (time x ygrid x xgrid)  Values range [0., 1.].
        seaice_conc_stdev:           Array of sea ice concentration stdev values like (time x ygrid x xgrid).  Values range [0., 1.].

        flag_missing:                Boolean array of missing data flags like (time x ygrid x xgrid).
        flag_land:                   Boolean array of land (land not adjacent to ocean) like (time x ygrid x xgrid).
        flag_coast:                  Boolean array of coast (land adjacent to ocean) flags like (time x ygrid x xgrid).
        flag_lake:                   Boolean array of lake data flags like (time x ygrid x xgrid).
        flag_hole:                   Boolean array of imaging hole flags like (time x ygrid x xgrid).

        latitude:                    Array of latitude coordinates as degrees north like (ygrid x xgrid).
        longitude:                   Array of longitude coordinates as degrees east like (ygrid x xgrid).
        x:                           Array of x-offsets in meters of the center of each cell from the projection center like (xgrid).
        y:                           Array of y-offsets in meters of the center of each cell from the projection center like (ygrid).
    """
    date: npt.NDArray[np.datetime64]

    seaice_conc: npt.NDArray[np.float64]
    seaice_conc_stdev: npt.NDArray[np.float64]

    flag_missing: npt.NDArray[np.bool]
    flag_land: npt.NDArray[np.bool]
    flag_coast: npt.NDArray[np.bool]
    flag_lake: npt.NDArray[np.bool]
    flag_hole: npt.NDArray[np.bool]

    latitude: npt.NDArray[np.float64]
    longitude: npt.NDArray[np.float64]

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]

    def __init__(self, nc_files: List[str] | List[Path]) -> None:
        """
        Initialize sea ice data in NOAA/NSIDC version 6 format.

        Args:
            nc_files (list of str or list of pathlib.Path):     .nc file to be opened
        """
        if len(nc_files) < 1: raise ValueError('Received an empty list of files!')

        self._nc_files = nc_files

        date = []
        seaice_conc, seaice_conc_stdev = [], []
        flag_missing, flag_land, flag_coast, flag_lake, flag_hole = [], [], [], [], []
        latitude, longitude = [], []
        x, y = [], []
        for f in self._nc_files:
            with netCDF4.Dataset(f) as d:
                date.append(np.array(d.variables['time']).astype('datetime64[D]'))

                s = np.array(d['cdr_seaice_conc'])
                flag_missing.append(s == 255)  # get flags
                flag_land.append(s == 254)
                flag_coast.append(s == 253)
                flag_lake.append(s == 252)
                flag_hole.append(s == 251)
                s[s >= 251] = np.nan  # mask out flags
                seaice_conc.append(s)

                s = np.array(d['cdr_seaice_conc_stdev'])
                s[s == -1] = np.nan  # mask out missing data
                seaice_conc_stdev.append(s)

                # handle everything else
                latitude.append(np.array(d['cdr_supplementary/latitude']))
                longitude.append(np.array(d['cdr_supplementary/longitude']))
                x.append(np.array(d['x']))
                y.append(np.array(d['y']))

        # check if all the grids match up
        for i in range(1, len(latitude)):
            if not np.all(np.isclose(latitude[i], latitude[0])) or \
               not np.all(np.isclose(longitude[i], longitude[0])) or \
               not np.all(np.isclose(x[i], x[0])) or \
               not np.all(np.isclose(y[i], y[0])):
                raise ValueError('Grid changed between files!  Cannot proceed!')

        # assign attributes
        self.date = np.concatenate(date)

        self.seaice_conc = np.concatenate(seaice_conc).transpose((0, 2, 1))
        self.seaice_conc_stdev = np.concatenate(seaice_conc_stdev).transpose((0, 2, 1))

        self.flag_missing = np.concatenate(flag_missing).transpose((0, 2, 1))
        self.flag_land = np.concatenate(flag_land).transpose((0, 2, 1))
        self.flag_coast = np.concatenate(flag_coast).transpose((0, 2, 1))
        self.flag_lake = np.concatenate(flag_lake).transpose((0, 2, 1))
        self.flag_hole = np.concatenate(flag_hole).transpose((0, 2, 1))

        self.latitude = latitude[0].T
        self.longitude = longitude[0].T
        self.x = x[0]
        self.y = y[0]


def check_boundaries(indices: List[int] | Tuple[int] | npt.NDArray[np.intp], d: SeaIceV6) -> None:
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
