import argparse
import datetime
import json
import re
import types

import neuralpde


DEFAULT_T = 7
DEFAULT_OFFSET = -1
DEFAULT_Q = 100
DEFAULT_KX = 5
DEFAULT_KS = 10
DEFAULT_REGION = (64, 110, 174, 235)
DEFAULT_WEIGHTS = (1., 1., 5., 5., 5., 5.)
DEFAULT_BATCH_SIZE = 200
DEFAULT_SHUFFLE = 10
DEFAULT_LR = 1e-3



_token_patterns = {
    int: r'(-?\d+)',
    float: r'(-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)',
}


def _parse_tokens(value: str, expected_type: type[int] | type[float], nargs: int):
    """Parse numeric tokens from user input."""
    raw_matches = re.findall(_token_patterns[expected_type] * nargs, value)
    if len(raw_matches) < nargs:
        raise ValueError(f'Expected {nargs} values of type {expected_type.__name__}')

    parsed = [expected_type(token) for token in raw_matches]
    return parsed[0] if nargs == 1 else parsed


def _validate_prompt(prompt: str, default=None):
    """Validate that the prompt string contains a {default} placeholder if a default value is provided."""
    if len(prompt) == 0:
        raise ValueError('Prompt cannot be empty.')
    
    if default is not None and '{default}' not in prompt:
        raise ValueError('Prompt must contain "{default}" placeholder when a default value is provided.')
    
    if len(prompt.format(default=default)) > 80:
        raise ValueError('Prompt (with default if default exists) cannot be longer than 80 characters.')


def get_option(prompt: str, expected_type: type[int] | type[float] | type[str], help: str = '', nargs: int = 1, default=None):
    """Get a typed user input value with optional default and nargs support."""
    _validate_prompt(prompt, default)

    while True:
        if help != '':
            print(help)
    
        raw = input('{:<81s}'.format(prompt if default is None else prompt.format(default=default)))
        print()

        if raw == '' and default is not None:
            return default

        if expected_type is str:
            return raw

        try:
            return _parse_tokens(raw, expected_type, nargs)
        except (ValueError, KeyError):
            print(f'Invalid input! Expected {nargs} values of type {expected_type.__name__}.')


if __name__ == '__main__':
    params = types.SimpleNamespace()

    models = {i + 1: s for i, s in enumerate(list(neuralpde.network.networks))}
    model = get_option('Model number:', int, help='Model to train:' + ''.join([f'\n\t{i}: {s}' for i, s in models.items()]))  
    if model not in models:
        print(f'Invalid model number!  Expected one of {list(models.keys())}.')
    params.model = models[model]

    params.timesteps = get_option('Number of timesteps to use in the network counting backward (default {default}):', int, default=DEFAULT_T)
    params.timestep_offset = get_option('The forward offset for timesteps to use in the network (default {default}):', int, default=DEFAULT_OFFSET)
    params.stages = get_option('Number of Runge-Kutta stages to use in integrating solution (default {default}):', int, default=DEFAULT_Q)
    params.kernel_x = get_option('Spatial kernel size (default {default}):', int, default=DEFAULT_KX)
    params.kernel_s = get_option('Number of neurons per output feature in deep neural network (default {default}):', int, default=DEFAULT_KS)
    params.region = get_option('Region (default {default}):', int,
                               help='Array indices to view, ordered as (left, right, up, down).\n' \
                                    'Note that this is in array coordinates and does not translate to cardinal directions.',
                               nargs=4, default=DEFAULT_REGION)
    params.weights = get_option('Weights of the loss terms (default {default}):', float, nargs=6, default=DEFAULT_WEIGHTS)
    params.lr = get_option('Learning rate of the optimizer (default {default}):', float, default=DEFAULT_LR)
    params.save = get_option('Path to save model (usually ending with .pth; enter <blank> for auto):', str)

    dest = get_option('Path to save this config (enter <blank> for auto):', str)

    with open(dest if dest else f'config-{params.model}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.json', 'w') as f:
        json.dump(vars(params), f, indent=4)
