"""
Module containing the objects, methods, and routines to ingest and present
usable sea ice concentration data from NOAA/NSIDC sea ice concentration
datafiles.
"""

import datetime
import netCDF4
import numpy as np
import numpy.typing as npt

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple



def date2datetime64(dt: datetime.date) -> np.datetime64:
    return np.datetime64(dt, 'D')


def lonlat2cartesian(longitude, latitude):
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    phi = np.deg2rad(longitude)
    theta = np.deg2rad(90 - latitude)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


@dataclass(eq=False, repr=False)
class SeaIceV6:
    """
    NOAA/NSIDC version 6 sea ice data class.

    This class includes the methods to ingest NOAA/NSIDC version 6 sea ice
    concentraton files.  See [this link](https://nsidc.org/data/g02202/versions/6)
    for more information.

    .. note::
        These data are on a polar stereographic grid according to the NSIDC Sea Ice Polar Stereographic Grid definition.

        See [this link](https://nsidc.org/data/polar-stereo/ps_grids.html) for more information.

    Attributes:
        date:                        Array of dates.
    
        seaice_conc:                 Array of shape (time, x, y) of fractional sea ice concentration values.  Values range [0., 1.].
        seaice_conc_stdev:           Array of shape (time, x, y) of sea ice concentration stdev values.  Values range [0., 1.].

        flag_missing:                Boolean array of shape (time, x, y) of missing data flags.
        flag_land:                   Boolean array of shape (time, x, y) of land (land not adjacent to ocean).
        flag_coast:                  Boolean array of shape (time, x, y) of coast (land adjacent to ocean) flags.
        flag_lake:                   Boolean array of shape (time, x, y) of lake data flags.
        flag_hole:                   Boolean array of shape (time, x, y) of imaging hole flags.

        latitude:                    Array of shape (x, y) of latitude coordinates as degrees north.
        longitude:                   Array of shape (x, y) of longitude coordinates as degrees east.
        x:                           Array of shape (x,) of x-offsets in meters of the center of each cell from the projection center.
        y:                           Array of shape (y,) of y-offsets in meters of the center of each cell from the projection center.
    """
    nc_files: InitVar[Sequence[str | Path]]

    date: npt.NDArray[np.datetime64] = field(init=False)

    seaice_conc: npt.NDArray[np.float64] = field(init=False)
    seaice_conc_stdev: npt.NDArray[np.float64] = field(init=False)

    flag_missing: npt.NDArray[np.bool_] = field(init=False)
    flag_land: npt.NDArray[np.bool_] = field(init=False)
    flag_coast: npt.NDArray[np.bool_] = field(init=False)
    flag_lake: npt.NDArray[np.bool_] = field(init=False)
    flag_hole: npt.NDArray[np.bool_] = field(init=False)

    latitude: npt.NDArray[np.float64] = field(init=False)
    longitude: npt.NDArray[np.float64] = field(init=False)

    x: npt.NDArray[np.float64] = field(init=False)
    y: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self, nc_files: Sequence[str | Path]) -> None:
        """
        Initialize sea ice data in NOAA/NSIDC version 6 format.

        Args:
            nc_files (list of str or list of pathlib.Path):     .nc file to be opened
        """
        if len(nc_files) < 1:
            raise ValueError('Received an empty list of files!')

        # TODO: cleanup the code below
        # consider abstracting reading individual files into separate method
        # also break lines of list defs for readability
        date = []
        seaice_conc, seaice_conc_stdev = [], []
        flag_missing, flag_land, flag_coast, flag_lake, flag_hole = [], [], [], [], []
        latitude, longitude = [], []
        x, y = [], []
        for f in nc_files:
            with netCDF4.Dataset(f) as d:
                date.append(np.array(d.variables['time']).astype('datetime64[D]'))

                s = np.asarray(d['cdr_seaice_conc'])
                flag_missing.append(s == 255)  # get flags
                flag_land.append(s == 254)
                flag_coast.append(s == 253)
                flag_lake.append(s == 252)
                flag_hole.append(s == 251)
                s[s >= 251] = np.nan  # mask out flags
                seaice_conc.append(s)

                s = np.asarray(d['cdr_seaice_conc_stdev'])
                s[s == -1] = np.nan  # mask out missing data
                seaice_conc_stdev.append(s)

                # handle everything else
                latitude.append(np.asarray(d['cdr_supplementary/latitude']))
                longitude.append(np.asarray(d['cdr_supplementary/longitude']))
                x.append(np.asarray(d['x']))
                y.append(np.asarray(d['y']))

        # check if all the grids match up
        for i in range(1, len(latitude)):
            if not np.all(np.isclose(latitude[i], latitude[0])) or \
               not np.all(np.isclose(longitude[i], longitude[0])) or \
               not np.all(np.isclose(x[i], x[0])) or \
               not np.all(np.isclose(y[i], y[0])):
                raise ValueError('Grid changed between files!  Cannot proceed!')

        # assign attributes
        self.date = np.concatenate(date)

        self.seaice_conc = np.concatenate(seaice_conc).transpose((0, 2, 1))  # transpose to (time, x, y) ordering
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
    Verify that boundaries at each index in `indices` are the same (i.e., boundaries are constant) and fails if not.

    Args:
        indices:        List or tuple of indices to check, (you probably want these to be adjacent).
        d:              Sea ice data object.
    """ 
    indices = list(indices)   
    if not np.all(d.flag_missing[indices[0]] == d.flag_missing[indices]) or \
        not np.all(d.flag_land[indices[0]] == d.flag_land[indices]) or \
        not np.all(d.flag_coast[indices[0]] == d.flag_coast[indices]) or \
        not np.all(d.flag_lake[indices[0]] == d.flag_lake[indices]) or \
        not np.all(d.flag_hole[indices[0]] == d.flag_hole[indices]):
        raise ValueError('Found inconsistent boundaries in data!')
