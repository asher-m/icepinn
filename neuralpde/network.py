import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from pathlib import Path

from . import layer


def get_rk_scheme(q: int = 100) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get an implicit Runge-Kutta scheme with `q` stages.

    Default of 100 because similar problems seem to use about the same, judging from a
    cursory perusing of github/maziarraissa/PINNs and github/rezaakb/pinns-torch.

    Arguments:
        q (int):
            Integer number of stages.
    """
    d = np.loadtxt(Path(__file__).parent / f'../raissi-2019/Utilities/IRK_weights/Butcher_IRK{q}.txt').astype(np.float64)
    a = np2torch(d[:q**2].reshape((q, q)))
    b = np2torch(d[q**2: q**2 + q])
    c = np2torch(d[q**2 + q:])

    return a, b, c


def np2torch(d: npt.NDArray) -> torch.Tensor:
    """
    Export numpy data to torch in every meaningful way, including sending it to the
    compute accelerator and casting it to the appropriate datatype.

    Arguments:
        d (array):
            Numpy array to export.
        dtype (torch.dtype):
            Datatype to which to cast the array.  Default is DTYPE (torch.float64 unless overridden).
    """
    return torch.from_numpy(d)


def torch2np(d: torch.Tensor) -> np.ndarray:
    """
    Export torch data back to numpy.

    Arguments:
        d (torch.Tensor):
            Torch tensor to export.
    """
    return d.numpy(force=True)


class AdrNondim:
    u0: nn.Buffer
    """Normalization constant u0."""
    L0: nn.Buffer
    """Normalization constant L0."""
    t0: nn.Buffer
    """Normalization constant t0."""
    k0: nn.Buffer
    """Normalization constant k0."""
    v0: nn.Buffer
    """Normalization constant v0."""
    f0: nn.Buffer
    """Normalization constant f0."""

    def __init__(
        self,
        u0: npt.NDArray[np.floating] | float,
        L0: npt.NDArray[np.floating] | float,
        t0: npt.NDArray[np.floating] | float,
        k0: npt.NDArray[np.floating] | float,
        v0: npt.NDArray[np.floating] | float,
        f0: npt.NDArray[np.floating] | float
    ):
        """
        Initialize ADR equation non-dimensionalization constants.

        Arguments:
            u0:     Scalar solution/data normalization.
            L0:     Scalar length-scale normalization.
            t0:     Scalar time-scale normalization.
            k0:     Scalar diffusivity normalization.
            v0:     Scalar velocity normalization.
            f0:     Scalar forcing normalization.

        .. note::
            These parameters can be unintuitive and challenging to guess.  Either use the numbers
            computed in the notebooks alongside this package or be careful when recomputing these values.
        """
        super().__init__()

        self.u0 = nn.Buffer(torch.tensor(u0))
        self.L0 = nn.Buffer(torch.tensor(L0))
        self.t0 = nn.Buffer(torch.tensor(t0))
        self.k0 = nn.Buffer(torch.tensor(k0))
        self.v0 = nn.Buffer(torch.tensor(v0))
        self.f0 = nn.Buffer(torch.tensor(f0))


class SeaIceAdr_x5_5k3_t3_w32_d3(AdrNondim, nn.Module):
    s: nn.Buffer
    """Time delta (i.e., :math:`dt`)."""
    h: nn.Buffer
    """Spatial delta (e.g., :math:`dx` or :math:`dy`)."""

    loss_weights: nn.Buffer
    """Weights of components of loss function."""

    k_avg: nn.Buffer
    """Local average kernel."""
    k_x: nn.Buffer
    """Finite difference kernel for first derivative in x."""
    k_y: nn.Buffer
    """Finite difference kernel for first derivative in y."""
    k_xx: nn.Buffer
    """Finite difference kernel for second derivative in x."""
    k_yy: nn.Buffer
    """Finite difference kernel for second derivative in y."""

    mix_a: nn.Parameter
    """Space-time mixer matrix."""
    mix_b: nn.Parameter
    """Space-time bias vector."""

    mlp: nn.Sequential
    """MLP stack on top of space-time convolution."""
    final: nn.Module
    """Mapping from final hidden layer to output."""

    def __init__(self, 
        u0: npt.NDArray[np.floating] | float,
        L0: npt.NDArray[np.floating] | float,
        t0: npt.NDArray[np.floating] | float,
        k0: npt.NDArray[np.floating] | float,
        v0: npt.NDArray[np.floating] | float,
        f0: npt.NDArray[np.floating] | float,
        s: npt.NDArray[np.floating] | float,
        h: npt.NDArray[np.floating] | float
    ) -> None:
        """
        Arguments:
            u0:     Solution/data normalization.
            L0:     Length-scale normalization.
            t0:     Time-scale normalization.
            k0:     Diffusivity normalization.
            v0:     Velocity normalization.
            f0:     Forcing normalization.
            s:          Time delta (i.e., :math:`dt`).
            h:          Spatial delta (e.g., :math:`dx` or :math:`dy`).
        """
        super().__init__(u0, L0, t0, k0, v0, f0)

        self.s = nn.Buffer(torch.tensor(s))
        self.h = nn.Buffer(torch.tensor(h))

        loss_weights = torch.tensor(
            [
                3,  # physics
                3,  # data
                1,  # parameter magnitude
                2,  # parameter smoothness
            ]
        )
        self.loss_weights = nn.Buffer(
            loss_weights / torch.sqrt(torch.sum(loss_weights ** 2))
        )

        # TODO: Consider constrained family of finite difference kernels.
        self.k_avg = nn.Buffer(
            torch.tensor(
                [  # normalized negative exponential weighting of points about query point
                    [0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.]
                ]
            )
        )
        self.k_x = nn.Buffer(
            0.5 / self.h * torch.tensor(
                [
                    [0., -1., 0.],
                    [0.,  0., 0.],
                    [0.,  1., 0.]
                ]
            )
        )
        self.k_y = nn.Buffer(
            0.5 / self.h * torch.tensor(
                [
                    [ 0., 0., 0.],
                    [-1., 0., 1.],
                    [ 0., 0., 0.]
                ]
            )
        )
        self.k_xx = nn.Buffer(
            1 / self.h ** 2 * torch.tensor(
                [
                    [0.,  1., 0.],
                    [0., -2., 0.],
                    [0.,  1., 0.]
                ]
            )
        )
        self.k_yy = nn.Buffer(
            1 / self.h ** 2 * torch.tensor(
                [
                    [0.,  0., 0.],
                    [1., -2., 1.],
                    [0.,  0., 0.]
                ]
            )
        )

        self.mix_a = nn.Parameter(torch.normal(0, 1, (32, 5 * 3 * 81)))  # 128 output size
        self.mix_b = nn.Parameter(torch.normal(0, 1, (32,)))

        self.mlp = nn.Sequential(
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
        )
        self.final = nn.Linear(32, 5)

    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, **kwargs):
        self.load_state_dict(torch.load(path), **kwargs)

    def forward(
            self,
            data: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the neural network posterior of the coordinates and data about the point (t, x, y).

        Accepts torch tensors as inputs.

        Arguments:
            data:       Tensor of shape (B, C, N, M) of DIMENSIONLESS data values about (t, x, y).
        where B is batch size, number of time steps C = 3, number of x and y values (N, M) = (11, 11).

        C index increments upwards for latter times.

        Returns a tensor.
        """
        b, c, n, m = data.shape
        if c != 3:
            raise ValueError(f'Expected C = 3 timesteps, got {c}!')
        if n != 11 or m != 11:
            raise ValueError(f'Expected number of x values, y values (N, M) = (11, 11), got {(n, m)}!')

        patches = nn.functional.unfold(data, 3).reshape((b, c, 3, 3, 81))
        patches_avg = torch.einsum('bcijl,ij->bcl', patches, self.k_avg).reshape((b, -1))
        patches_x = torch.einsum('bcijl,ij->bcl', patches, self.k_x).reshape((b, -1))
        patches_y = torch.einsum('bcijl,ij->bcl', patches, self.k_y).reshape((b, -1))
        patches_xx = torch.einsum('bcijl,ij->bcl', patches, self.k_xx).reshape((b, -1))
        patches_yy = torch.einsum('bcijl,ij->bcl', patches, self.k_yy).reshape((b, -1))

        mixed = torch.einsum(
            'bp,ip->bi',
            torch.cat((patches_avg, patches_x, patches_y, patches_xx, patches_yy), dim=1),
            self.mix_a
        ) + self.mix_b
        hidden = self.mlp(mixed)
        output = self.final(hidden)

        return output

    def fit(
        self,
        data: torch.Tensor,
        label: torch.Tensor
    ):
        """
        Compute fit of network to data.

        Accepts torch tensors as inputs.

        Arguments:
            data:       Tensor of shape (B, C, N, M) of data values about (t, x, y).
            label:      Tensor of shape (B,) of labels.
        where B is batch size, number of time steps C = 3, number of x values N = 13 and number of y values M = 13.

        C index increments upwards for latter times.  Note that (N, M) = (13, 13); this is so the solution can
        computed at additional locations surrouding the query point so derivatives can be approximated by finite
        differences.

        C index increments upwards for latter times.
        """
        b, c, n, m = data.shape
        if c != 3:
            raise ValueError(f'Expected C = 3 timesteps, got {c}!')
        if n != 13 or m != 13:
            raise ValueError(f'Expected number of x values, y values (N, M) = (13, 13), got {(n, m)}!')
        
        if label.shape != (b,):
            raise ValueError(f'Expected labels of shape (B,) = ({b},), got {label.shape}!')

        patches = nn.functional.unfold(data, 11).reshape(b, 3, 11, 11, 9)
        patches = patches.permute(0, 4, 1, 2, 3).reshape(b * 9, 3, 11, 11)
        outputs = self.forward(patches).reshape(b, 3, 3, 5)

        loss = self._compute_loss(data, label, outputs)

        return loss

    def _compute_loss(
        self,
        data: torch.Tensor,
        label: torch.Tensor,
        outputs: torch.Tensor,
    ):
        """
        Compute loss of outputs.

        Accepts torch tensors as inputs.

        Arguments:
            data:               Tensor of shape (B, C, N, M) of data values about (t, x, y).
            label:              Tensor of shape (B,) of labels.
            outputs:            Tensor of shape (B, 3, 3, O) of values.
        where B is batch size, number of time steps C = 3, number of x and y values (N, M) = (11, 11),
        and number of output channels O = 5.

        C index increments upwards for latter times.

        Returns a tensor.
        """
        b, c, n, m = data.shape
        if c != 3:
            raise ValueError(f'Expected C = 3 timesteps, got {c}!')
        if n != 13 or m != 13:
            raise ValueError(f'Expected number of x values, y values (N, M) = (13, 13), got {(n, m)}!')

        ob, on, om, oo = outputs.shape
        if ob != b:
            raise ValueError(f'Expected output batch size B = {b}, got {ob}!')
        if on != 3 or om != 3:
            raise ValueError(f'Expected number of output x values, y values (N, M) = (3, 3), got {(on, om)}!')
        if oo != 5:
            raise ValueError(f'Expected output channels O = 5, got {oo}!')

        (
            _,
            kappa,
            velx,
            vely,
            force
        ) = self._unpack_outputs(outputs[:, 1, 1, :])
        (
            _,
            kappa_x,
            velx_x,
            _,
            force_x
        ) = self._unpack_outputs(torch.einsum('bnmo,nm->bo', outputs, self.k_x))
        (
            _,
            kappa_y,
            _,
            vely_y,
            force_y
        ) = self._unpack_outputs(torch.einsum('bnmo,nm->bo', outputs, self.k_y))
        (
            _,
            _,
            _,
            _,
            _
        ) = self._unpack_outputs(torch.einsum('bnmo,nm->bo', outputs, self.k_xx))
        (
            _,
            _,
            _,
            _,
            _
        ) = self._unpack_outputs(torch.einsum('bnmo,nm->bo', outputs, self.k_yy))

        soln = data[:, 2, 6, 6]
        soln_x = torch.einsum('bnm,nm->b', data[:, 2, 5:8, 5:8], self.k_x)
        soln_y = torch.einsum('bnm,nm->b', data[:, 2, 5:8, 5:8], self.k_y)
        soln_xx = torch.einsum('bnm,nm->b', data[:, 2, 5:8, 5:8], self.k_xx)
        soln_yy = torch.einsum('bnm,nm->b', data[:, 2, 5:8, 5:8], self.k_yy)

        soln_t = (
            self.k0 * self.t0 / self.L0 ** 2 * (kappa_x * soln_x + kappa_y * soln_y + kappa * (soln_xx + soln_yy)) +
            self.v0 * self.t0 / self.L0 * (soln * (velx_x + vely_y) + soln_x * velx + soln_y * vely) * -1 +
            self.f0 * self.t0 / self.u0 * force
        )

        l_physics = (
            label - (soln + soln_t * self.s)
        ) ** 2
        l_data = (
            soln - label
        ) ** 2
        l_norm = (  # normalization of params
            kappa ** 2 +
            velx ** 2 +
            vely ** 2 +
            force ** 2
        )
        l_norm_diff = (  # normalization of smoothness of params
            velx_x ** 2 +
            vely_y ** 2 +
            kappa_x ** 2 +
            kappa_y ** 2 +
            force_x ** 2 +
            force_y ** 2
        )

        l = torch.stack(
            (
                l_physics,
                l_data,
                l_norm,
                l_norm_diff
            ),
            dim=-1
        # ) @ self.loss_weights
        )

        return l

    def _unpack_outputs(
        self,
        outputs: torch.Tensor,
    ):
        """
        Unpack a outputs from :meth:`_forward`.

        .. seealso::
            See the complementary method :meth:`_pack_outputs`.

        Accepts torch tensors as inputs.

        Arguments:
            outputs:      Tensor of shape (B, ..., O) of output values.
        where B is batch size and number of output channels O = 5.

        Returns a tuple of tensors.
        """
        soln = outputs[..., 0]
        kappa = outputs[..., 1]
        velx = outputs[..., 2]
        vely = outputs[..., 3]
        force = outputs[..., 4]
        return soln, kappa, velx, vely, force

    def _pack_outputs(
        self,
        soln: torch.Tensor,
        kappa: torch.Tensor,
        velx: torch.Tensor,
        vely: torch.Tensor,
        force: torch.Tensor,
    ):
        """
        Pack output values as if from :meth:`_forward`.

        .. seealso::
            See the complementary method :meth:`_unpack_outputs`.

        Accepts torch tensors as inputs.

        Arguments:
            soln:        Tensor of shape (B, ...) of solution.
            kappa:       Tensor of shape (B, ...) of kappa.
            velx:        Tensor of shape (B, ...) of vx.
            vely:        Tensor of shape (B, ...) of vy.
            force:       Tensor of shape (B, ...) of f.
        where B is batch size.
            
        Returns a tensor of shape (B, ..., 5).
        """
        return torch.stack(
            (
                soln,
                kappa,
                velx,
                vely,
                force,
            ),
            dim=-1
        )
