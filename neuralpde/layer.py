import math
import torch
import torch.nn as nn
import warnings

from functools import reduce
from typing import Tuple



class LocallyConnected2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            in_spatial_shape: Tuple[int],
            kernel_size: int,
            dilation: int = 1,
            padding: int = 0,
            stride: int = 1,
    ):
        """
        Initializes a locally-connected 2d layer.

        This is a simple implementation that I can expand later if I want or need.

        See [the PyTorch docs on `torch.nn.Unfold`](https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html)
        for a more thorough description of these parameters and how they work.  In particular, I've tried
        to reuse the notational convention described in that documentation here.

        Note: Currently, only 4-D input tensors (batched image-like tensors) are supported,
        as per `nn.Unfold` and `nn.functional.unfold`.

        Args:
            in_channels         : Number of channels in the input image (e.g., 3 for RGB).
            out_channels        : Number of output channels (number of filters per spatial location).
            in_spatial_shape    : Input spatial shape (spatial_1 x spatial_2 x ...).
            kernel_size         : The size of the sliding window.
            dilation            : Stride of elements within a kernel neighborhood.
            stride              : Stride for the sliding window.
            padding             : Zero-padding added to both sides of the input.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_spatial_shape = in_spatial_shape
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

        self.out_spatial_size = tuple(
            (
                (self.in_spatial_shape[d] + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
                for d in range(len(self.in_spatial_shape))
            )
        )
        """ Output spatial shape """

        # Total number of blocks from unfold (unused, but also helped me write this in the first place)
        # I want to keep this here to prevent this from descending into the arcane
        # I.e., the process is important!
        self.L = reduce(lambda x, y: x * y, self.out_spatial_size)
        """ Total number of blocks """

        self._block_size = self.in_channels * self.kernel_size**len(self.in_spatial_shape)
        """ Number of elements in each block """

        self.weight = nn.Parameter(torch.ones(out_channels, *self.out_spatial_size, self._block_size) / self._block_size)
        self.bias = nn.Parameter(torch.randn(out_channels, *self.out_spatial_size))

    def forward(self, x: torch.Tensor):
        """
        Note: Currently, only 4-D input tensors (batched image-like tensors) are supported,
        as per `nn.Unfold` and `nn.functional.unfold`.  This method will add a degenerate axis
        at position 0 if it receives a tensor of ndim = 3, otherwise it will throw an error.

        Args:
            x (Tensor): Input tensor with shape (batch_size, in_channels, *self.in_spatial_shape)

        Returns:
            out (Tensor): Output tensor with shape (batch_size, out_channels, *self.out_spatial_shape)
        """
        if x.ndim == 4:
            batch_size = x.size(0)

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_spatial_size))
            return out

        elif x.ndim == 3:
            x = x.unsqueeze(0)
            batch_size = 1

            blocks = torch.nn.functional.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
            out = torch.einsum('bil,oli->bol', blocks, self.weight.flatten(1, -2)) + self.bias.flatten(1, -1).unsqueeze(0)
            out = out.reshape((batch_size, self.out_channels, *self.out_spatial_size))
            return out.squeeze(0)

        else:
            raise NotImplementedError(
                'Only 4-D (batched image-like tensors) or 3-D tensors (unbatched image-like tensors) are supported!\n'
                'nn.Unfold and nn.functional.unfold only support 4-D tensors.'
            )


class GaussianDistanceWeight(nn.Module):
    def __init__(
            self,
            *coordinates: Tuple[torch.Tensor]
    ):
        """
        Compute the Gaussian weight across an array of coordinates to a point x.  Outputs
        an array of weights.

        Args:
            coordinates:        Tuple of 1-D tensors of coordinate ranges.  Coordinate ranges are usually
                                monotonic, but I don't check and I don't think it will break anything.

        Every tuple in *coordinates should have the same number of coordinate ranges (i.e., the same spatial dimension,)
        and corresponding coordinate ranges should have the same length.  For example, if coordinates = ((x1, y1), (x2, y2)),
        then len(x1) = len(x2) and len(y1) = len(y2).
        """
        super().__init__()

        if len(coordinates) == 1:
            self.coordinates = nn.Buffer(
                torch.stack(torch.meshgrid(*coordinates[0], indexing='ij'))
            )
            self.batched = False
        elif len(coordinates) > 1:
            self.coordinates = nn.Buffer(
                torch.stack([torch.stack(torch.meshgrid(*c, indexing='ij')) for c in coordinates])
            )
            self.batched = True
        else:
            raise ValueError('Received no coordinate ranges!')

        self.dim = len(coordinates[0])
        """ Number of spatial dimensions. """

        self.width = nn.Parameter(torch.ones(1))
        """ Width of the Gaussian. """

    def forward(
            self,
            x: torch.Tensor
    ):
        """
        Args:
            x:      A tensor where the last dimension corresponds to the number of dimensions
                    (i.e. coordinate ranges) used at instance initialization.

        If batched (i.e., multiple coordinate ranges were used at instance initialization,) the first axis of the input
        tensor must correspond to the number of batches (coordinate ranges or number of tuples) pass at initialization.
        """
        assert x.shape[-1] == self.dim, f'Received x with incompatible number of dimensions (x.shape[-1] = {x.shape[-1]}, expected {self.dim})!s'
        if self.batched:
            assert len(x) == len(self.coordinates), f'Received x with incompatible batch number (len(x) = {len(x)}, expected {len(self.coordinates)})!'
            dsq = torch.sum((self.coordinates[:, *(None,) * (x.ndim - 2), ...] - x[:, ..., *(None,) * self.dim])**2, -3)
        else:
            dsq = torch.sum((self.coordinates[*(None,) * (x.ndim - 1), ...] - x[..., *(None,) * self.dim])**2, -3)
        return torch.exp(-1/2 * dsq / self.width**2)


class NReLU(nn.Module):
    def __init__(self, n: int = 2, inplace=False):
        r"""
        A ReLU to the n'th power.  Nice because it's computationally fast, has unbounded
        activation, and (n-1)-times continuously differentiable.

        This also normalizes by n factorial so that the (n-1)'th and n'th derivatives match with
        ReLU and the Heaviside, respectively.

        Arguments:
            n:          Power you to which you'd like to raise ReLU.  Also corresponds to the
                        (n-1)-differentiability of this activation function.

        Let typical ReLU $r$ and NReLU $r^n / n!$.  Note $r \in C^0(\mathbb{R})$, specifically,
        $$
            r' = \begin{cases}
                0 & x \leq 0 \\
                1 & x    > 0
            \end{cases}
        $$
        Realize, of course, that $r'$ is not continuous.  On the other hand, $r^n / n! \in C^{n-1}(\mathbb{R})$, as,
        $$
            (r^n)' = \begin{cases}
                0                & x \leq 0 \\
                x^{n-1} / (n-1)! & x    > 0
            \end{cases}
        $$
        """
        if n > 4: warnings.warn(f"Received n = {n}!  This is big and might cause floating point "
                                "problems.  Make sure you know what you're doing!")

        super().__init__()
        self.n = n
        self.n_factorial = math.factorial(n)
        self.ReLU = nn.ReLU(inplace)

    def forward(self, x):
        return self.ReLU(x)**self.n / self.n_factorial
