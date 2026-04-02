import math
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from pathlib import Path
from typing import List, Tuple

from . import layer


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.xpu.is_available():
    DEVICE = 'xpu'
else:
    DEVICE = 'cpu'
DTYPE = torch.float32

# can I think of DT as unitless, or do I need to consider its units more carefully?
DT = 1



def normalize_xy(x: np.ndarray, y: np.ndarray) -> Tuple[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Normalize spatial coordinates (from meters to unitless dimension on the interval [-1, 1]).

    Returns a tuple of ((scalex, scaley), (x_normalized, y_normalized)).
    """
    assert x.ndim == 1 and y.ndim == 1, "I don't know how to handle multi-D arrays!"
    scalex, scaley = np.ptp(x), np.ptp(y)
    return (scalex, scaley), ((x - np.mean(x)) / scalex, (y - np.mean(y)) / scaley)


def normalize_data(u: np.ndarray):
    raise ValueError('Sea ice data already normalized!')


def get_rk_scheme(q: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get an implicit Runge-Kutta scheme with `q` stages.

    Default of 100 because similar problems seem to use about the same, judging from a
    cursory perusing of github/maziarraissa/PINNs and github/rezaakb/pinns-torch.

    Arguments:
        q (int):
            Integer number of stages.
    """
    d = np.loadtxt(Path(__file__).parent / f'../raissi-2019/Utilities/IRK_weights/Butcher_IRK{q}.txt').astype(np.float32)
    A = np2torch(d[:q**2].reshape((q, q)))
    b = np2torch(d[q**2: q**2 + q])
    c = np2torch(d[q**2 + q:])

    return A, b, c


def np2torch(d: npt.NDArray, dtype: torch.dtype = DTYPE) -> torch.Tensor:
    """
    Export numpy data to torch in every meaningful way, including sending it to the
    compute accelerator and casting it to the appropriate datatype.

    Arguments:
        d (array):
            Numpy array to export.
        dtype (torch.dtype):
            Datatype to which to cast the array.  Default is DTYPE (torch.float32 unless overridden).
    """
    return torch.from_numpy(d).to(DEVICE, dtype)


def torch2np(d: torch.Tensor) -> np.ndarray:
    """
    Export torch data back to numpy.

    Arguments:
        d (torch.Tensor):
            Torch tensor to export.
    """
    return d.detach().cpu().numpy()


class SeaiceAdr(nn.Module):
    """Abstract class implementing the ADR equation for sea ice coverage data."""
    q: int
    """Number of stages q used in Runge-Kutta scheme to compute loss.  See Raissi 2019."""
    rk_a: torch.Tensor
    """Runge-Kutta A matrix."""
    rk_b: torch.Tensor
    """Runge-Kutta b vector."""
    rk_c: torch.Tensor
    """Runge-Kutta c vector."""
    channels: int
    """Number of output channels of the network, like (k, v1, v2, f, rk_1, rk_2, ..., rk_q)."""

    def __init__(self, q: int):
        """Initialize the PINN.
        
        Arguments:
            q (int):
                Number of stages q used in RK scheme to compute loss.  See Raissi 2019.
        """
        super().__init__()
        self.q = q
        self.rk_a, self.rk_b, self.rk_c = map(nn.Buffer, get_rk_scheme(q))
        self.channels = 4 + q

    def predict(self, xd: np.ndarray, yd: np.ndarray, batch_size: int = 100):
        """Push data through the network for evaluation.

        Arguments:
            xd (np.ndarray):
                1-D array of x coordinates of collocation points at which to evaluate the solution.
            yd (np.ndarray):
                1-D array of y coordiantes of collocation points at which to evaluate the solution.
            batch_size (int):
                Number of collocation points to evaluate simultaneously; choose this to be as large as your GPU's memory will allow.
        """
        self.eval()

        results = {
            'k': list(),
            'v1': list(),
            'v2': list(),
            'f': list(),
            'uhat_i': list(),
            'uhat_f': list(),
        }

        indices = np2torch(np.indices((len(xd), len(yd))).reshape((2, -1)), dtype=torch.int)

        # TODO: remove y as seperate coordinate array
        # make x of shape (N, d), where N number of points to evaluate and d spatial dimension of domain
        # same as changes to fit
        x = np2torch(xd).requires_grad_(True)
        y = np2torch(yd).requires_grad_(True)

        for i in range(math.ceil(indices.shape[-1] / batch_size)):
            x = x[indices[0, i * batch_size:(i + 1) * batch_size]]
            y = y[indices[1, i * batch_size:(i + 1) * batch_size]]

            diff_r = torch.ones((len(x), self.channels), device=DEVICE).requires_grad_(True)
            diff_p = torch.ones((len(x)), device=DEVICE).requires_grad_(True)

            r = self.forward(x, y)
            r_x, = torch.autograd.grad(torch.autograd.grad(r, x, diff_r, create_graph=True), diff_r, diff_p, create_graph=True)
            r_xx, = torch.autograd.grad(torch.autograd.grad(r_x, x, diff_r, create_graph=True), diff_r, diff_p, retain_graph=True)
            r_y, = torch.autograd.grad(torch.autograd.grad(r, y, diff_r, create_graph=True), diff_r, diff_p, create_graph=True)
            r_yy, = torch.autograd.grad(torch.autograd.grad(r_y, y, diff_r, create_graph=True), diff_r, diff_p, retain_graph=False)

            k, v1, v2, f, urk = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4:]
            k_x, v1_x, v2_x, urk_x = r_x[..., 0], r_x[..., 1], r_x[..., 2], r_x[..., 4:]
            k_y, v1_y, v2_y, urk_y = r_y[..., 0], r_y[..., 1], r_y[..., 2], r_y[..., 4:]
            urk_xx = r_xx[..., 4:]
            urk_yy = r_yy[..., 4:]

            pde = (  # not particularly pythonic, but easier to read
                k.unsqueeze(-1) * (urk_xx + urk_yy) + (k_x.unsqueeze(-1) * urk_x + k_y.unsqueeze(-1) * urk_y)
                - (v1_x.unsqueeze(-1) + v2_y.unsqueeze(-1)) * urk - (v1.unsqueeze(-1) * urk_x + v2.unsqueeze(-1) * urk_y)
                + f.unsqueeze(-1)
            )

            uhat_i = urk - DT * torch.einsum('ij,bj->bi', self.rk_a, pde)
            uhat_f = urk - DT * torch.einsum('ij,bj->bi', self.rk_a - self.rk_b.unsqueeze(0), pde)

            results['k'].append(torch2np(k))
            results['v1'].append(torch2np(v1))
            results['v2'].append(torch2np(v2))
            results['f'].append(torch2np(f))
            results['uhat_i'].append(torch2np(uhat_i))
            results['uhat_f'].append(torch2np(uhat_f))

        # TODO: fix linting error about overwriting values in results (expected to be lists)
        for key in ('k', 'v1', 'v2', 'f'):
            results[key] = np.concatenate(results[key]).reshape((len(x), len(y)))
        for key in ('uhat_i', 'uhat_f'):
            results[key] = np.concatenate(results[key]).reshape((len(x), len(y), self.q))

        return results


    def fit(self,
            u: npt.NDArray,
            x_range: npt.NDArray,
            y_range: npt.NDArray,
            dmask_inte: npt.NDArray,
            dmask_peri: npt.NDArray,
            weights: npt.NDArray,
            epochs: int = 1000,
            lr: float = 1e-3,
            batch_size: int = 100,
            shuffle: int = 10):
        """Train the PINN.

        Arguments:
            u:              Solution data of each cell at t_{n} and t_{n+1}.  Must be of shape (2, N, M).
            x_range:        The x coordinate range as a 1-D array of length N.
            y_range:        The y coordinate range as a 1-D array of length M.
            dmask_inte:     Mask of the interior of the domain in which the PDE will be enforced (and interior loss terms.)  Must be of shape (N, M).
            dmask_peri:     Mask of the perimeter of the domain on which the boundary conditions will be enforced (and boundary loss terms.)  Must be of shape (N, M).
            weights:        Weights of each term in the loss.  Must be of shape (6,) and ordered as (loss_u_i, loss_u_f, loss_bc, loss_k_reg, loss_v_reg, loss_f_min).
            epochs:         Number of epochs to run.
            lr:             Learning rate passed to Adam optimizer.
            batch_size:     Number of solution points on which to simultaneously train.
            shuffle:        Shuffle the set of solution points after this many epochs.
        """
        weights = np2torch(weights)
        mask_inte, mask_peri = np2torch(dmask_inte, dtype=bool), np2torch(dmask_peri, dtype=bool)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # cast ndarrays to tensors
        u = np2torch(u)
        # TODO: remove y as seperate coordinate array
        # make x of shape (N, d), where N number of points to evaluate and d spatial dimension of domain
        # same as changes to predict
        x_range = np2torch(x_range)
        y_range = np2torch(y_range)

        # make indices
        indices = (mask_inte | mask_peri).nonzero().T

        # make right-hand vectors for Jacobian computation
        diff_r = torch.ones((batch_size, self.channels), device=DEVICE).requires_grad_(True)
        diff_p = torch.ones((batch_size), device=DEVICE).requires_grad_(True)

        # do the training
        self.train()
        losses = list()
        t = Path('training')  # training lockfile
        t.touch()
        e = 0
        # TODO: this is pretty nasty; need to break down
        while t.exists() and e < epochs + 1:
            print(f'Starting epoch {e}...', end='\r')

            optimizer.zero_grad(set_to_none=True)

            if e % shuffle == 0:  # shuffle the set of training points
                idx = torch.randint(0, indices.shape[-1], (batch_size,), device=DEVICE)
                x = x_range[indices[0, idx]].requires_grad_(True)
                y = y_range[indices[1, idx]].requires_grad_(True)
                u_i = u[0, indices[0, idx], indices[1, idx]]
                u_f = u[1, indices[0, idx], indices[1, idx]]
                mi = mask_inte[indices[0, idx], indices[1, idx]]
                mb = mask_peri[indices[0, idx], indices[1, idx]]

            r = self.forward(x, y)
            r_x, = torch.autograd.grad(torch.autograd.grad(r, x, diff_r, create_graph=True), diff_r, diff_p, create_graph=True)
            r_xx, = torch.autograd.grad(torch.autograd.grad(r_x, x, diff_r, create_graph=True), diff_r, diff_p, create_graph=True)
            r_y, = torch.autograd.grad(torch.autograd.grad(r, y, diff_r, create_graph=True), diff_r, diff_p, create_graph=True)
            r_yy, = torch.autograd.grad(torch.autograd.grad(r_y, y, diff_r, create_graph=True), diff_r, diff_p, create_graph=False)

            k, v1, v2, f, urk = r[..., 0], r[..., 1], r[..., 2], r[..., 3], r[..., 4:]
            k_x, v1_x, v2_x, urk_x = r_x[..., 0], r_x[..., 1], r_x[..., 2], r_x[..., 4:]
            k_y, v1_y, v2_y, urk_y = r_y[..., 0], r_y[..., 1], r_y[..., 2], r_y[..., 4:]
            urk_xx = r_xx[..., 4:]
            urk_yy = r_yy[..., 4:]

            pde = (  # not particularly pythonic, but easier to read
                k.unsqueeze(-1) * (urk_xx + urk_yy) + (k_x.unsqueeze(-1) * urk_x + k_y.unsqueeze(-1) * urk_y)
                - (v1_x.unsqueeze(-1) + v2_y.unsqueeze(-1)) * urk - (v1.unsqueeze(-1) * urk_x + v2.unsqueeze(-1) * urk_y)
                + f.unsqueeze(-1)
            )

            uhat_i = urk - DT * torch.einsum('ij,bj->bi', self.rk_a, pde)
            uhat_f = urk - DT * torch.einsum('ij,bj->bi', self.rk_a - self.rk_b.unsqueeze(0), pde)

            # compute loss with estimate and actual solution
            loss_u_i = torch.sum(mi.unsqueeze(-1) * (uhat_i - u_i.unsqueeze(-1))**2)
            loss_u_f = torch.sum(mi.unsqueeze(-1) * (uhat_f - u_f.unsqueeze(-1))**2)

            # compute other loss terms
            loss_bc = torch.sum(mb * (v1**2 + v2**2)) + torch.sum(mb * k**2)
            loss_k_reg = torch.sum(mi * (k_x**2 + k_y**2)) + torch.sum(mi * k**2)
            loss_v_reg = torch.sum(mi * (v1_x**2 + v1_y**2 + v2_x**2 + v2_y**2))
            loss_f_min = torch.sum(mi * (f**2))

            # compute net loss
            loss = torch.stack((loss_u_i, loss_u_f, loss_bc, loss_k_reg, loss_v_reg, loss_f_min)) @ weights

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if e % 10 == 0:
                print(
                    f'Epoch {e:5d} of {epochs:5d}, loss {losses[-1]:10.2f}' +
                    (f', relative improvement {100 * (1 - losses[-1] / losses[-11]):10.2f}%' if e > 0 else '')
                )

            del r, r_x, r_y, r_xx, r_yy
            e += 1

        t.unlink(missing_ok=True)  # remove lockfile

        return losses


class DiscConvPinn(SeaiceAdr):
    def __init__(
            self,
            q: int,
            Nt: int,
            x_range: np.ndarray,
            y_range: np.ndarray,
            kernel_xy: int,
            kernel_stack: int
    ) -> None:
        """
        Initialize the PINN.

        Arguments:
            q:                  Number of stages q used in RK scheme to compute loss.  See Raissi 2019.
            Nt:                 Number of solution maps (i.e., in time) included in buffer.  The network is conditioned on these known solutions.
            x_range:            The x coordinate range as a 1-D array.
            y_range:            The y coordinate range as a 1-D array.
            kernel_xy:          Size of kernel over which to convolve.
            kernel_stack:       Number of neurons in the FC stack per output channel.
        Nt must be the same for every batch, data for which must be loaded through `Network.load_data`.
        """
        super().__init__()

        self.q = q
        self.Nt = Nt
        self.dx = np.diff(x_range)[0]
        assert np.all(np.isclose(np.diff(x_range), self.dx)), 'Received irregularly shaped x_range!'
        self.dy = np.diff(y_range)[0]
        assert np.all(np.isclose(np.diff(y_range), self.dy)), 'Received irregularly shaped y_range!'
        self.kernel = kernel_xy
        assert kernel_xy % 2 == 1, 'Kernel size must be odd!'

        # make kernel offsets
        self.offsets_xy = nn.Buffer(
            np2torch(
                ((np.indices((kernel_xy, kernel_xy)) - kernel_xy // 2) * \
                 np.array((self.dx, self.dy))[:, None, None]).transpose((1, 2, 0))
            )
        )

        self.rk_A, self.rk_b, self.rk_c = map(nn.Buffer, get_rk_scheme(q))

        self.channels = 4 + q  # parameters + rk stages
        self.spatial_correlation = layer.GaussianDistanceWeight(  # recreate ranges to handle padding
            (
                torch.linspace(x_range[0] - kernel_xy//2 * self.dx, x_range[-1] + kernel_xy//2 * self.dx, len(x_range) + kernel_xy - 1),
                torch.linspace(y_range[0] - kernel_xy//2 * self.dy, y_range[-1] + kernel_xy//2 * self.dy, len(y_range) + kernel_xy - 1)
            )
        )
        nn.init.constant_(self.spatial_correlation.width, min(abs(self.dx), abs(self.dy)) / 1.5)
        self.padding = nn.ReflectionPad2d(kernel_xy//2)
        self.layers = nn.Sequential(
            nn.Linear(Nt * kernel_xy**2, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels * kernel_stack),
            nn.ReLU(),
            nn.Linear(self.channels * kernel_stack, self.channels)
        )


    def save(self, path: str | Path):
        torch.save(self.state_dict(), path)


    def load(self, path: str | Path, **kwargs):
        self.load_state_dict(torch.load(path), **kwargs)


    def data_(
            self,
            u: np.ndarray,
    ) -> None:
        """
        Load sea ice data into the network.

        Arguments:
            u:      Data conditioning the model.
        """
        if u.ndim == 4:
            data = np2torch(u).contiguous()
        elif u.ndim == 3:  # i.e., batch_size = 1
            data = np2torch(u).unsqueeze(0).contiguous()
        else:
            raise ValueError('Receieved data of an incompatible shape.  Check docstring!')
        self.data = nn.Buffer(self.padding(data))


    def forward(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            batch_num: int = 0,
    ) -> torch.Tensor:
        """
        Push data through the network for training.

        Arguments:
            x:          1-D tensor of x-values at which to evaluate the solution of length batch size.
            y:          1-D tensor of y-values at which to evaluate the solution of length batch size.
            batch_num:  Index of self.data to use.  Only used when x and y are scalars.

        Returns a tensor on `DEVICE`.
        """
        if x.ndim > 0:  # non-scalar case
            r = self.spatial_correlation(torch.stack((x, y))[:, None, None, :] + self.offsets_xy[None, ...])
            r = r[:, None, ...] * self.data[:, :, None, None, ...]
            r = torch.sum(r, dim=(-1, -2))
            r = r.flatten(1, -1)
            r = self.layers(r)
            return r
        else:
            r = self.spatial_correlation(torch.stack((x, y))[None, None, :] + self.offsets_xy)
            r = r[None, ...] * self.data[batch_num, :, None, None, ...]
            r = torch.sum(r, dim=(-1, -2))
            r = r.flatten(0, -1)
            r = self.layers(r)
            return r


class AttentionPinn(SeaiceAdr):
    def __init__(self, q: int):
        raise NotImplementedError('Not implemented yet!')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Not implemented yet!')


networks = {'DiscConvPinn': DiscConvPinn,
            'AttentionPinn': AttentionPinn}
