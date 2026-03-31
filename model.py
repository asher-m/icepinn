import argparse
import datetime
import matplotlib.colors
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pickle

from dateutil import parser as dateparser
from typing import List, Tuple, Dict, Callable

import neuralpde



RESAMPLE = 1000
""" Number of times to train on each data point.  Used to calculate number of epochs from batch size. """

DEFAULTS_T = 2
DEFAULTS_OFFSET = 0
DEFAULTS_Q = 4
DEFAULTS_KXY = 5
DEFAULTS_KS = 10
DEFAULTS_REGION = (64, 110, 174, 235)
DEFAULTS_WEIGHTS = (1., 1., 5., 5., 5., 5.)
DEFAULTS_BATCH_SIZE = 200
DEFAULTS_SHUFFLE = 10
DEFAULTS_LR = 1e-3

def cmap_transparent(cmap: str, transform: Callable, n: int = 256):
    colors = plt.get_cmap(cmap)(np.arange(n))
    colors[:, -1] = transform(np.linspace(0, 1, n))
    return matplotlib.colors.ListedColormap(colors)

CMAP_ICE = plt.get_cmap('Blues_r')
CMAP_ICE.set_bad(color='tan')
CMAP_ERR = cmap_transparent('RdYlGn_r', np.vectorize(lambda x: (5/3 * x) if x <= 0.3 else ((1 - 0.5) / (1 - 0.3) * (x - 1) + 1)))
CMAP_ERR.set_bad(color='tan')
CMAP_ERR.set_over(color='darkmagenta')
CMAP_PARAM = plt.get_cmap('jet')
CMAP_PARAM.set_bad(color='tan')



def vrange(a: np.ndarray, vmin=None) -> Dict:
    extreme = np.nanmax(np.abs(a))
    return {'vmin': -extreme if vmin is None else vmin, 'vmax': extreme}


def normalize_weights(w: np.ndarray) -> np.ndarray:
    return w / np.sqrt(np.sum(w**2))


def hide(a, mask):
    return np.ma.masked_where(mask, a)


def plot(x_scale: float, y_scale: float,
         x_range: np.ndarray, y_range: np.ndarray,
         mask_interior: np.ndarray, mask_boundary: np.ndarray,
         solution: np.ndarray, r: Dict, show=True) -> matplotlib.figure.Figure | None:
    # convert xscale, yscale to km
    x_scale = x_scale / 1e3
    y_scale = y_scale / 1e3

    mask = ~mask_interior | mask_boundary

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20, 10))

    ax = axes[0, 0]
    d = hide(solution[0], mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T, cmap=CMAP_ICE, vmin=0., vmax=1.)
    c = fig.colorbar(p, ax=ax)
    c.set_label('Fractional Sea Ice Concentration')
    ax.set_ylabel('Distance (km)')

    ax = axes[0, 1]
    d1 = hide(np.mean(r['uhat_i'], -1), mask)
    d2 = hide(np.abs(solution[0] - np.mean(r['uhat_i'], -1)), mask)
    ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                  d1.T, vmin=0., vmax=1., cmap=CMAP_ICE)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d2.T, cmap=CMAP_ERR, vmin=0., vmax=0.3)
    c = fig.colorbar(p, ax=ax, extend='max')
    c.set_label('Absolute Error')

    ax = axes[0, 2]
    d = hide(solution[1], mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T, cmap=CMAP_ICE, vmin=0., vmax=1.)
    c = fig.colorbar(p, ax=ax)
    c.set_label('Fractional Sea Ice Concentration')

    ax = axes[0, 3]
    d1 = hide(np.mean(r['uhat_f'], -1), mask)
    d2 = hide(np.abs(solution[1] - np.mean(r['uhat_f'], -1)), mask)
    ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                  d1.T, vmin=0., vmax=1., cmap=CMAP_ICE)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d2.T, cmap=CMAP_ERR, vmin=0., vmax=0.3)
    c = fig.colorbar(p, ax=ax, extend='max')
    c.set_label('Absolute Error')

    ax = axes[1, 0]
    d = hide(r['k'] * x_scale * y_scale / 24, mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T , cmap=CMAP_PARAM)
    c = fig.colorbar(p, ax=ax)
    c.set_label('Diffusivity\n(km² / hour)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')

    ax = axes[1, 1]
    d = hide(r['v1'] * x_scale / 24, mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T, cmap=CMAP_PARAM, **vrange(d))
    c = fig.colorbar(p, ax=ax)
    c.set_label('Lateral Velocity\n(km / hour)')
    ax.set_xlabel('Distance (km)')

    ax = axes[1, 2]
    d = hide(r['v2'] * y_scale / 24, mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T, cmap=CMAP_PARAM, **vrange(d))
    c = fig.colorbar(p, ax=ax)
    c.set_label('Vertical Velcoity\n(km / hour)')
    ax.set_xlabel('Distance (km)')

    ax = axes[1, 3]
    d = hide(r['f'] / 24, mask)
    p = ax.pcolormesh(x_range * x_scale, y_range * y_scale,
                      d.T, cmap=CMAP_PARAM, **vrange(d))
    c = fig.colorbar(p, ax=ax)
    c.set_label('Forcing\n(fractional sea ice per hour)')
    ax.set_xlabel('Distance (km)')

    for ax in axes.flatten():
        ax.set_aspect('equal')

    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig


def main(date,
         timesteps: int = DEFAULTS_T,
         offset: int = DEFAULTS_OFFSET,
         q: int = DEFAULTS_Q,
         kernel_xy: int = DEFAULTS_KXY,
         kernel_stack: int = DEFAULTS_KS,
         region: Tuple[int] | List[int] = DEFAULTS_REGION,
         weights: Tuple[int] | List[int] = DEFAULTS_WEIGHTS,
         batch_size: int = DEFAULTS_BATCH_SIZE,
         epochs: int | None = None,
         shuffle: int = DEFAULTS_SHUFFLE,
         lr: float = DEFAULTS_LR,
         save: str | None = None,
         load: str | None = None,
         init_save: str | None = None,
         init_load: str | None = None) -> Dict:
    if init_load:
        with open(init_load, 'rb') as fp:
            p = pickle.load(fp)
        date = p['date']
        region = p['region']
        timesteps = p['timesteps']
        offset = p['offset']
        solution = p['solution']
        window = p['window']
        mask_i = p['mask_i']
        mask_p = p['mask_p']
        q = p['q']
        x_scale = p['x_scale']
        y_scale = p['y_scale']
        x_range = p['x_range']
        y_range = p['y_range']
        kernel_xy = p['kernel_xy']
        kernel_stack = p['kernel_stack']
    else:
        date = dateparser.parse(date).date()
        region = np.s_[region[0]:region[1], region[2]:region[3]]
        years = set(map(lambda d: d.year,
                        (date,
                        date + datetime.timedelta(days=1),
                        *(date + datetime.timedelta(days=offset + t) for t in range(timesteps)))))
        data = neuralpde.nc.SeaIceV6(sorted([f'data/V4/seaice_conc_daily_nh_{y}_v04r00.nc' for y in iter(years)]))
        date_idx = np.searchsorted(data.date, date)
        neuralpde.nc.check_boundaries([date_idx, date_idx + 1] + [date_idx + offset + t for t in range(timesteps)], data)

        solution = data.seaice_conc[date_idx:date_idx + 2, *region]
        solution[np.isnan(solution)] = 0.
        window = data.seaice_conc[date_idx + offset:date_idx + offset + timesteps, *region]
        window[np.isnan(window)] = 0.

        (x_scale, y_scale), (x_range, y_range) = neuralpde.network.normalize_xy(data.meters_x[region[0]], data.meters_y[region[1]])
        mask_i = ~(data.flag_missing | data.flag_land | data.flag_lake | data.flag_hole)[date_idx, *region]  # interior mask
        mask_p = data.flag_coast[date_idx, *region]  # perimeter mask

    weights = normalize_weights(np.array(weights))

    if init_save:
        with open(init_save, 'wb') as fp:
            pickle.dump(
                {
                    'date': date,
                    'region': region,
                    'timesteps': timesteps,
                    'offset': offset,
                    'solution': solution,
                    'window': window,
                    'mask_i': mask_i,
                    'mask_p': mask_p,
                    'q': q,
                    'x_scale': x_scale,
                    'y_scale': y_scale,
                    'x_range': x_range,
                    'y_range': y_range,
                    'kernel_xy': kernel_xy,
                    'kernel_stack': kernel_stack},
                fp
            )

    net = neuralpde.network.Network(q, timesteps, x_range, y_range, kernel_xy, kernel_stack)
    net = net.to(neuralpde.network.DEVICE, neuralpde.network.DTYPE)
    net.data_(window)

    if load:
        net.load(load)
    else:
        net.fit(solution, x_range, y_range,
                mask_i, mask_p,
                weights=weights, batch_size=batch_size,
                epochs=epochs if epochs else len(x_range) * len(y_range) // batch_size * RESAMPLE,
                shuffle=shuffle, lr=lr)

    if save: net.save(save)

    r = net.predict(x_range, y_range, batch_size)
    plot(x_scale, y_scale, x_range, y_range, mask_i, mask_p, solution, r)

    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('date',
                        help='The date from which to extract ice flow PDE parameters.  ' \
                        'Can be in any format `dateutil.parser` can understand, (e.g., '
                        'YYYYMMDD).', type=str)
    parser.add_argument('-t', '--timesteps',
                        help='Number of timesteps against which to convolve to feed to ' \
                        'the deep neural network.',
                        type=int, default=DEFAULTS_T, dest='timesteps')
    parser.add_argument('-o', '--timestep-offset',
                        help='The first timestep to (relative to `date`) to convolve ' \
                        'into the network.',
                        type=int, default=DEFAULTS_OFFSET, dest='offset')
    parser.add_argument('-q', '--stages',
                        help='Number of RK stages to use in integrating solution.  ' \
                        'Newton\'s method is q=1, higher order schemes are Gauss-Legendre ' \
                        'RK methods.',
                        type=int, default=DEFAULTS_Q, dest='q')
    parser.add_argument('-kxy', '--kernel-xy',
                        help='Spatial kernel size.  Precisely, the number of spatial ' \
                        'kernels (in one dimension) against which to convolve to feed ' \
                        'to the deep neural network.',
                        type=int, default=DEFAULTS_KXY, dest='kernel_xy')
    parser.add_argument('-ks', '--kernel-stack',
                        help='Deep neural network kernel size.  Precisely, the number ' \
                        'of neurons per output feature (q + 4 for q RK stages and 4 PDE ' \
                        'parameters).',
                        type=int, default=DEFAULTS_KS, dest='kernel_stack')
    parser.add_argument('-r', '--region',
                        help='Array indices to view, ordered as `left right up down` '
                        '(note that this list is not comma-separated).  Defaults to ' \
                        'the region north of (right of) Barrow, Alaska.  Note that this ' \
                        'argument is in array coordinates and does not translate to ' \
                        'cardinal directions.',
                        type=int, nargs=4, default=DEFAULTS_REGION, dest='region')
    parser.add_argument('-w', '--weights',
                        help='Weight vector of the loss terms in order of `(estimate ' \
                        'at t_n, estimate at t_n+1, boundary, kappa normalization, v ' \
                        'normalization, f normalization)`.  This vector is normalized ' \
                        'such that the scaling does not matter, only the scale of any ' \
                        'term relative to the others.',
                        type=float, nargs=6, default=DEFAULTS_WEIGHTS, dest='weights')
    parser.add_argument('-b', '--batch-size',
                        help='Number of points on which to train simultaneously.  ' \
                        'Significiantly impacts memory usage.',
                        type=int, default=DEFAULTS_BATCH_SIZE, dest='batch_size')
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs for which to train the model.',
                        type=int, dest='epochs')
    parser.add_argument('-s', '--shuffle',
                        help='Number of epochs after which the training points are shuffled.',
                        type=int, default=DEFAULTS_SHUFFLE, dest='shuffle')
    parser.add_argument('-l', '--learning-rate',
                        help='Learning rate of the optimizer.',
                        type=float, default=DEFAULTS_LR, dest='lr')
    parser.add_argument('--save',
                        help='Path (usually ending with `.pth`) where to save model ' \
                        'parameters.',
                        type=str, dest='save')
    parser.add_argument('--load',
                        help='Path (usually ending with `.pth`) from where to load ' \
                        'model parameters.  This will override training the model, ' \
                        '(i.e., it is assumed these parameters are of a trained model.)  ' \
                        'NOTE: this script must be called with the same parameters used ' \
                        'to create the model! Pass defaults if defaults were used to ' \
                        'create the model.', type=str, dest='load')
    parser.add_argument('--init-save',
                        help='Path where to save parameters used to create the model.  ' \
                        'This path will be written as a pickle binary object.',
                        type=str, dest='init_save')
    parser.add_argument('--init-load',
                        help='Path from where to load parameters used to create the model.',
                        type=str, dest='init_load')
    main(**vars(parser.parse_args()))
