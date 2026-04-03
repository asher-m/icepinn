import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from typing import Tuple

from . import layer


if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.xpu.is_available():
    DEVICE = 'xpu'
else:
    DEVICE = 'cpu'
DTYPE = torch.float32

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

    def _split_output_channels(
            self,
            outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split the output channels of the network into diffusivity, velocity_x, velocity_y, forcing, and stage values."""
        diffusivity = outputs[..., 0]
        velocity_x = outputs[..., 1]
        velocity_y = outputs[..., 2]
        forcing = outputs[..., 3]
        stage_values = outputs[..., 4:]
        return diffusivity, velocity_x, velocity_y, forcing, stage_values

    def _compute_output_derivatives(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            create_graph: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the output of the network and its first and second derivatives with respect to x and y at the given points."""
        # TODO: Benchmark replacing these nested autograd.grad calls with torch.func.jacfwd/vmap.
        # This network has only 2 differentiated inputs (x, y) but many output channels, so
        # forward-mode AD may be faster for these first- and second-derivative calculations.
        batch_size = len(x)
        diff_outputs = torch.ones((batch_size, self.channels), device=DEVICE).requires_grad_(True)
        diff_points = torch.ones((batch_size,), device=DEVICE).requires_grad_(True)

        outputs = self.forward(x, y)
        outputs_x, = torch.autograd.grad(
            torch.autograd.grad(outputs, x, diff_outputs, create_graph=True),
            diff_outputs,
            diff_points,
            create_graph=True,
        )
        outputs_xx, = torch.autograd.grad(
            torch.autograd.grad(outputs_x, x, diff_outputs, create_graph=True),
            diff_outputs,
            diff_points,
            create_graph=create_graph,
            retain_graph=True,
        )
        outputs_y, = torch.autograd.grad(
            torch.autograd.grad(outputs, y, diff_outputs, create_graph=True),
            diff_outputs,
            diff_points,
            create_graph=True,
        )
        outputs_yy, = torch.autograd.grad(
            torch.autograd.grad(outputs_y, y, diff_outputs, create_graph=True),
            diff_outputs,
            diff_points,
            create_graph=False,
            retain_graph=False,
        )

        return outputs, outputs_x, outputs_y, outputs_xx, outputs_yy

    def _compute_pde_rhs(
            self,
            outputs: torch.Tensor,
            outputs_x: torch.Tensor,
            outputs_y: torch.Tensor,
            outputs_xx: torch.Tensor,
            outputs_yy: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the right-hand side of the PDE at the given points, using the outputs and their derivatives."""
        diffusivity, velocity_x, velocity_y, forcing, stage_values = self._split_output_channels(outputs)
        diffusivity_x, velocity_x_x, _, _, stage_values_x = self._split_output_channels(outputs_x)
        diffusivity_y, _, velocity_y_y, _, stage_values_y = self._split_output_channels(outputs_y)
        _, _, _, _, stage_values_xx = self._split_output_channels(outputs_xx)
        _, _, _, _, stage_values_yy = self._split_output_channels(outputs_yy)

        return (
            diffusivity.unsqueeze(-1) * (stage_values_xx + stage_values_yy)
            + (diffusivity_x.unsqueeze(-1) * stage_values_x + diffusivity_y.unsqueeze(-1) * stage_values_y)
            - (velocity_x_x.unsqueeze(-1) + velocity_y_y.unsqueeze(-1)) * stage_values
            - (velocity_x.unsqueeze(-1) * stage_values_x + velocity_y.unsqueeze(-1) * stage_values_y)
            + forcing.unsqueeze(-1)
        )

    def _reconstruct_endpoint_estimates(
            self,
            stage_values: torch.Tensor,
            pde_rhs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct estimates of the solution at the initial and final time steps from the stage values and PDE right-hand side."""
        u_hat_initial = stage_values - DT * torch.einsum('ij,bj->bi', self.rk_a, pde_rhs)
        u_hat_final = stage_values - DT * torch.einsum('ij,bj->bi', self.rk_a - self.rk_b.unsqueeze(0), pde_rhs)
        return u_hat_initial, u_hat_final

    def _get_training_batch(
            self,
            data: torch.Tensor,
            x: torch.Tensor,
            y: torch.Tensor,
            mask_interior: torch.Tensor,
            mask_boundary: torch.Tensor,
            mask: torch.Tensor,
            batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_indices = torch.randint(0, mask.shape[-1], (batch_size,), device=DEVICE)
        x_batch = x[mask[0, batch_indices]].requires_grad_(True)
        y_batch = y[mask[1, batch_indices]].requires_grad_(True)
        u_initial_batch = data[0, mask[0, batch_indices], mask[1, batch_indices]]
        u_final_batch = data[1, mask[0, batch_indices], mask[1, batch_indices]]
        interior_mask_batch = mask_interior[mask[0, batch_indices], mask[1, batch_indices]]
        boundary_mask_batch = mask_boundary[mask[0, batch_indices], mask[1, batch_indices]]
        return x_batch, y_batch, u_initial_batch, u_final_batch, interior_mask_batch, boundary_mask_batch

    def predict(
            self,
            x: np.ndarray,
            y: np.ndarray,
            batch_size: int = 100
        ) -> dict[str, np.ndarray]:
        """Infer a solution at the given collocation points, accepting numpy arrays as inputs.

        Arguments:
            xd:
                1-D array of x coordinates of collocation points at which to evaluate the solution.
            yd:
                1-D array of y coordiantes of collocation points at which to evaluate the solution.
            batch_size:
                Number of collocation points to evaluate simultaneously; choose this to be as large as your GPU's memory will allow.
                
        """
        return self._predict(
            np2torch(x),
            np2torch(y),
            batch_size
        )

    def _predict(
            self,
            x: torch.Tensor,
            y: torch.Tensor,
            batch_size: int
        ):
        """Infer a solution at the given collocation points, accepting torch tensors as inputs.

        Arguments:
            xd:
                1-D array of x coordinates of collocation points at which to evaluate the solution.
            yd:
                1-D array of y coordiantes of collocation points at which to evaluate the solution.
            batch_size:
                Number of collocation points to evaluate simultaneously; choose this to be as large as your GPU's memory will allow.
                
        """
        # TODO: Consider removing y as separate coordinate array; make x of shape (N, d)
        # where N number of points to evaluate and d spatial dimension of domain.
        # This change should mirror similar changes in associated methods.
        # This would complicate assumptions about the dimension/shape of convolutional kernels,
        # but would make the code generalizable to higher-dimensional problems and more consistent across methods.
        self.eval()
        num_x, num_y = len(x), len(y)

        results = {
            'k': list(),
            'v1': list(),
            'v2': list(),
            'f': list(),
            'uhat_i': list(),
            'uhat_f': list(),
        }

        indices = np2torch(np.indices((num_x, num_y)).reshape((2, -1)), dtype=torch.int)

        for batch_start in range(0, indices.shape[-1], batch_size):
            batch_stop = batch_start + batch_size
            x_batch = x[indices[0, batch_start:batch_stop]]
            y_batch = y[indices[1, batch_start:batch_stop]]

            outputs, outputs_x, outputs_y, outputs_xx, outputs_yy = self._compute_output_derivatives(
                x_batch,
                y_batch,
                create_graph=False,
            )
            diffusivity, velocity_x, velocity_y, forcing, stage_values = self._split_output_channels(outputs)
            pde_rhs = self._compute_pde_rhs(outputs, outputs_x, outputs_y, outputs_xx, outputs_yy)
            u_hat_initial, u_hat_final = self._reconstruct_endpoint_estimates(stage_values, pde_rhs)

            results['k'].append(torch2np(diffusivity))
            results['v1'].append(torch2np(velocity_x))
            results['v2'].append(torch2np(velocity_y))
            results['f'].append(torch2np(forcing))
            results['uhat_i'].append(torch2np(u_hat_initial))
            results['uhat_f'].append(torch2np(u_hat_final))

        for key in ('k', 'v1', 'v2', 'f'):
            results[key] = np.concatenate(results[key]).reshape((num_x, num_y))
        for key in ('uhat_i', 'uhat_f'):
            results[key] = np.concatenate(results[key]).reshape((num_x, num_y, self.q))

        return results

    def fit(
        self,
        data: npt.NDArray,
        x: npt.NDArray,
        y: npt.NDArray,
        mask_interior: npt.NDArray,
        mask_perimeter: npt.NDArray,
        weights: npt.NDArray,
        epochs: int = 1000,
        lr: float = 1e-3,
        batch_size: int = 100,
        shuffle: int = 10
    ) -> list[float]:
        """Train the PINN, accepting numpy arrays as inputs.

        Arguments:
            data:
                Solution data of each cell at t_{n} and t_{n+1}.  Must be of shape (2, N, M).
            x_range:
                The x coordinate range as a 1-D array of length N.
            y_range:
                The y coordinate range as a 1-D array of length M.
            mask_interior:
                Mask of the interior of the domain in which the PDE will be enforced (and interior loss terms.)  Must be of shape (N, M).
            mask_perimeter:
                Mask of the perimeter of the domain on which the boundary conditions will be enforced (and boundary loss terms.)  Must be of shape (N, M).
            weights:
                Weights of each term in the loss.  Must be of shape (6,) and ordered as (loss_u_i, loss_u_f, loss_bc, loss_k_reg, loss_v_reg, loss_f_min).
            epochs:
                Number of epochs to run.
            lr:
                Learning rate passed to Adam optimizer.
            batch_size:
                Number of solution points on which to simultaneously train.
            shuffle:
                Shuffle the set of solution points after this many epochs.
        """
        return self._fit(
            np2torch(data),
            np2torch(x),
            np2torch(y),
            np2torch(mask_interior),
            np2torch(mask_perimeter),
            np2torch(weights),
            epochs,
            lr,
            batch_size,
            shuffle
        )

    def _fit(
        self,
        data: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask_interior: torch.Tensor,
        mask_perimeter: torch.Tensor,
        weights: torch.Tensor,
        epochs: int,
        lr: float,
        batch_size: int,
        shuffle: int
    ):
        """Train the PINN, accepting torch tensors as inputs.

        Arguments:
            data:
                Solution data of each cell at t_{n} and t_{n+1}.  Must be of shape (2, N, M).
            x_range:
                The x coordinate range as a 1-D array of length N.
            y_range:
                The y coordinate range as a 1-D array of length M.
            mask_interior:
                Mask of the interior of the domain in which the PDE will be enforced (and interior loss terms.)  Must be of shape (N, M).
            mask_perimeter:
                Mask of the perimeter of the domain on which the boundary conditions will be enforced (and boundary loss terms.)  Must be of shape (N, M).
            weights:
                Weights of each term in the loss.  Must be of shape (6,) and ordered as (loss_u_i, loss_u_f, loss_bc, loss_k_reg, loss_v_reg, loss_f_min).
            epochs:
                Number of epochs to run.
            lr:
                Learning rate passed to Adam optimizer.
            batch_size:
                Number of solution points on which to simultaneously train.
            shuffle:
                Shuffle the set of solution points after this many epochs.
        """
        # TODO: Consider removing y as separate coordinate array; make x of shape (N, d)
        # where N number of points to evaluate and d spatial dimension of domain.
        # This change should mirror similar changes in associated methods.
        # This would complicate assumptions about the dimension/shape of convolutional kernels,
        # but would make the code generalizable to higher-dimensional problems and more consistent across methods.
        optimizer = optim.Adam(self.parameters(), lr=lr)

        mask = (mask_interior | mask_perimeter).nonzero().T

        # do the training
        self.train()
        losses = list()
        training_lockfile = Path('training')
        training_lockfile.touch()
        epoch = 0
        # TODO: Replace lockfile-based interruption with a more standard graceful stop
        # mechanism such as KeyboardInterrupt/SIGINT handling plus checkpoint cleanup.
        while training_lockfile.exists() and epoch < epochs + 1:
            print(f'Starting epoch {epoch}...', end='\r')

            optimizer.zero_grad(set_to_none=True)

            if epoch % shuffle == 0:  # shuffle the set of training points
                (
                    x_batch,
                    y_batch,
                    u_initial_batch,
                    u_final_batch,
                    mask_interior_batch,
                    mask_boundary_batch,
                ) = self._get_training_batch(
                    data,
                    x,
                    y,
                    mask_interior,
                    mask_perimeter,
                    mask,
                    batch_size,
                )

            outputs, outputs_x, outputs_y, outputs_xx, outputs_yy = self._compute_output_derivatives(
                x_batch,
                y_batch,
                create_graph=True,
            )
            diffusivity, velocity_x, velocity_y, forcing, stage_values = self._split_output_channels(outputs)
            diffusivity_x, velocity_x_x, velocity_y_x, _, _ = self._split_output_channels(outputs_x)
            diffusivity_y, velocity_x_y, velocity_y_y, _, _ = self._split_output_channels(outputs_y)
            pde_rhs = self._compute_pde_rhs(outputs, outputs_x, outputs_y, outputs_xx, outputs_yy)
            u_hat_initial, u_hat_final = self._reconstruct_endpoint_estimates(stage_values, pde_rhs)

            loss_u_i = torch.sum(mask_interior_batch.unsqueeze(-1) * (u_hat_initial - u_initial_batch.unsqueeze(-1))**2)
            loss_u_f = torch.sum(mask_interior_batch.unsqueeze(-1) * (u_hat_final - u_final_batch.unsqueeze(-1))**2)

            loss_bc = torch.sum(mask_boundary_batch * (velocity_x**2 + velocity_y**2)) + torch.sum(mask_boundary_batch * diffusivity**2)
            loss_k_reg = torch.sum(mask_interior_batch * (diffusivity_x**2 + diffusivity_y**2)) + torch.sum(mask_interior_batch * diffusivity**2)
            loss_v_reg = torch.sum(mask_interior_batch * (velocity_x_x**2 + velocity_x_y**2 + velocity_y_x**2 + velocity_y_y**2))
            loss_f_min = torch.sum(mask_interior_batch * (forcing**2))

            loss = torch.stack((loss_u_i, loss_u_f, loss_bc, loss_k_reg, loss_v_reg, loss_f_min)) @ weights

            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if epoch % 10 == 0:
                print(
                    f'Epoch {epoch:5d} of {epochs:5d}, loss {losses[-1]:10.2f}' +
                    (f', relative improvement {100 * (1 - losses[-1] / losses[-11]):10.2f}%' if epoch > 0 else '')
                )

            del outputs, outputs_x, outputs_y, outputs_xx, outputs_yy
            epoch += 1

        training_lockfile.unlink(missing_ok=True)

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
