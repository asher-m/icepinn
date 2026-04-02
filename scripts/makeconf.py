import argparse
import datetime
import json
import re
import types

import neuralpde


DEFAULT_T = 7
DEFAULT_OFFSET = -1
DEFAULT_Q = 4
DEFAULT_KX = 5
DEFAULT_KS = 10
DEFAULT_REGION = (64, 110, 174, 235)
DEFAULT_WEIGHTS = (1., 1., 5., 5., 5., 5.)
DEFAULT_BATCH_SIZE = 200
DEFAULT_SHUFFLE = 10
DEFAULT_LR = 1e-3



getopt_patterns = {
    int: r'(-?\d+)',
    float: r'(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)',
}
def getopt(prompt: str, type: type[int] | type[float] | type[str], nargs: int = 1, default=None):
    while True:
        p = input(prompt if default is None else prompt.format(default=default))

        if len(p) == 0 and default is not None:
            return default

        if type == str:
            return p
        else:
            m = re.findall(getopt_patterns[type] * nargs, p)
            if len(m) < nargs:
                print(f'Invalid input!  Expected {type} and nargs {nargs}.')
            else:
                if nargs == 1:
                    return type(m[0])
                else:
                    return [type(v) for v in m]


if __name__ == '__main__':
    params = types.SimpleNamespace()

    models = {i + 1: s for i, s in enumerate(list(neuralpde.network.networks))}
    model = getopt('Model to train:' + ''.join([f'\n\t{i}: {s}' for i, s in models.items()]) + '\nModel number: ', int)
    if model not in models:
        print(f'Invalid model number!  Expected one of {list(models.keys())}.')
    params.model = models[model]

    params.timesteps = getopt('Number of timesteps to use in the network counting backward (default {default}): ', int, default=DEFAULT_T)
    params.timestep_offset = getopt('The forward offset for timesteps to use in the network (default {default}): ', int, default=DEFAULT_OFFSET)
    params.stages = getopt('Number of Runge-Kutta stages to use in integrating solution (default {default}): ', int, default=DEFAULT_Q)
    params.kernel_x = getopt('Spatial size of convolutional kernel (symmetric in spatial dimensions; default {default}): ', int, default=DEFAULT_KX)
    params.kernel_s = getopt('Number of neurons per output feature in the deep neural network kernel (default {default}): ', int, default=DEFAULT_KS)
    params.region = getopt('Array indices to view, ordered as (left, right, up, down).\n' \
                           'Note that this argument is in array coordinates and does not translate to cardinal directions.\n' \
                           'Region (default {default}; enter without parens): ', int, nargs=4, default=DEFAULT_REGION)
    params.weights = getopt('Weights of the loss terms (default {default}; enter without parens): ', float, nargs=6, default=DEFAULT_WEIGHTS)
    params.lr = getopt('Learning rate of the optimizer (default {default}): ', float, default=DEFAULT_LR)
    params.save = getopt('Path where to save model (usually ending with .pth; enter nothing to generate automatically): ', str)

    dest = getopt('Path to save the config file generated from this script (enter nothing to generate automatically): ', str)

    with open(dest if dest else f'config-{params.model}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w') as f:
        json.dump(vars(params), f, indent=4)
