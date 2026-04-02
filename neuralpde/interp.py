import datetime
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.interpolate as interpolate
import scipy.spatial as spatial

import neuralpde


FIGSIZE = (7, 7)

cmap = plt.get_cmap('Blues_r')
cmap.set_bad(color='red', alpha=0.3)

d = neuralpde.nc.SeaIceV6(['sic_psn25_19790101-19791231_v06r00.nc'])


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


_d_hull_2d = spatial.ConvexHull(np.column_stack(lonlat2cartesian(d.longitude.flatten(), d.latitude.flatten())[:2]))
def d_hull_interior(longitude, latitude, tol=1e-12) -> npt.NDArray[np.bool]:
    s = longitude.shape
    return np.reshape(np.all(np.column_stack(lonlat2cartesian(longitude.flatten(), latitude.flatten())[:2]) @ _d_hull_2d.equations[:, :-1].T + _d_hull_2d.equations[:, -1:].T < tol, axis=1), s)

_d_interp = lambda longitude, latitude: interpolate.NearestNDInterpolator(np.column_stack(lonlat2cartesian(d.longitude.flatten(), d.latitude.flatten())),
                                                                          d.seaice_conc[0].flatten())(lonlat2cartesian(longitude, latitude))
def d_interp(longitude, latitude):
    longitude = np.asarray(longitude)
    latitude = np.asarray(latitude)

    a = _d_interp(longitude, latitude)
    a[~d_hull_interior(longitude, latitude)] = np.nan
    
    return a


fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, subplot_kw={'projection': 'polar'})
for i in range(len(d.longitude)):
    ax.plot(np.deg2rad(d.longitude[i]), 90-d.latitude[i])
llong = np.deg2rad(np.concat((d.longitude[0, :], d.longitude[:, -1], d.longitude[-1, ::-1], d.longitude[::-1, 0])))
llat = 90 - (np.concat((d.latitude[0, :], d.latitude[:, -1], d.latitude[-1, ::-1], d.latitude[::-1, 0])))
ax.plot(llong, llat, 'k-', linewidth=4)
plt.show()


llong, llat = np.meshgrid(np.linspace(0, 360, 500), np.linspace(40, 90, 500))
fig, ax = plt.subplots(1, 1, figsize=FIGSIZE, subplot_kw={'projection': 'polar'})
ax.pcolormesh(np.deg2rad(llong), 90 - llat, d_interp(llong, llat), cmap=cmap, rasterized=True)
plt.show()
