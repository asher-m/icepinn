import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neuralpde.network import DT, SeaiceAdr


class DummySeaiceAdr(SeaiceAdr):
    def __init__(self, q: int = 2):
        super().__init__(q)
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.stack((
            x + y,
            x - y,
            x * y,
            x**2 + y**2,
            x**2 + y,
            x + y**2,
        ), dim=-1)


# TODO: verify that this test is correct
class SeaiceAdrTests(unittest.TestCase):
    def test_predict_preserves_grid_shape_and_values_across_batches(self):
        model = DummySeaiceAdr(q=2)
        x_coords = np.array([-1.0, 0.5, 1.5], dtype=np.float32)
        y_coords = np.array([-2.0, 1.0], dtype=np.float32)

        predictions = model.predict(x_coords, y_coords, batch_size=2)

        expected_x, expected_y = np.meshgrid(x_coords, y_coords, indexing='ij')
        expected_k = expected_x + expected_y
        expected_v1 = expected_x - expected_y
        expected_v2 = expected_x * expected_y
        expected_f = expected_x**2 + expected_y**2
        stage_values = np.stack((expected_x**2 + expected_y, expected_x + expected_y**2), axis=-1)
        stage_x = np.stack((2 * expected_x, np.ones_like(expected_x)), axis=-1)
        stage_y = np.stack((np.ones_like(expected_y), 2 * expected_y), axis=-1)
        stage_xx = np.stack((2 * np.ones_like(expected_x), np.zeros_like(expected_x)), axis=-1)
        stage_yy = np.stack((np.zeros_like(expected_y), 2 * np.ones_like(expected_y)), axis=-1)
        pde_rhs = (
            expected_k[..., None] * (stage_xx + stage_yy)
            + np.stack((np.ones_like(expected_x), np.ones_like(expected_x)), axis=-1) * stage_x
            + np.stack((np.ones_like(expected_y), np.ones_like(expected_y)), axis=-1) * stage_y
            - (np.ones_like(expected_x) + expected_x)[..., None] * stage_values
            - (expected_v1[..., None] * stage_x + expected_v2[..., None] * stage_y)
            + expected_f[..., None]
        )
        expected_u_hat_initial = stage_values - DT * np.einsum(
            'ij,xyj->xyi',
            model.rk_a.detach().cpu().numpy(),
            pde_rhs
        )
        expected_u_hat_final = stage_values - DT * np.einsum(
            'ij,xyj->xyi',
            (model.rk_a - model.rk_b.unsqueeze(0)).detach().cpu().numpy(),
            pde_rhs,
        )

        self.assertEqual(predictions['k'].shape, (len(x_coords), len(y_coords)))
        self.assertEqual(predictions['v1'].shape, (len(x_coords), len(y_coords)))
        self.assertEqual(predictions['v2'].shape, (len(x_coords), len(y_coords)))
        self.assertEqual(predictions['f'].shape, (len(x_coords), len(y_coords)))
        self.assertEqual(predictions['uhat_i'].shape, (len(x_coords), len(y_coords), model.q))
        self.assertEqual(predictions['uhat_f'].shape, (len(x_coords), len(y_coords), model.q))

        np.testing.assert_allclose(predictions['k'], expected_k, atol=1e-5)
        np.testing.assert_allclose(predictions['v1'], expected_v1, atol=1e-5)
        np.testing.assert_allclose(predictions['v2'], expected_v2, atol=1e-5)
        np.testing.assert_allclose(predictions['f'], expected_f, atol=1e-5)
        np.testing.assert_allclose(predictions['uhat_i'], expected_u_hat_initial, atol=1e-5)
        np.testing.assert_allclose(predictions['uhat_f'], expected_u_hat_final, atol=1e-5)

    def test_fit_smoke_runs_with_shared_helpers(self):
        model = DummySeaiceAdr(q=2)
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        y = np.array([-0.5, 0.5], dtype=np.float32)
        data = np.zeros((2, len(x), len(y)), dtype=np.float32)
        mask_interior = np.ones((len(x), len(y)), dtype=bool)
        mask_boundary = np.zeros((len(x), len(y)), dtype=bool)
        weights = np.ones(6, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                losses = model.fit(
                    data,
                    x,
                    y,
                    mask_interior,
                    mask_boundary,
                    weights,
                    epochs=1,
                    lr=1e-3,
                    batch_size=3,
                    shuffle=1,
                )
            finally:
                os.chdir(previous_cwd)

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_minimal_subclass_only_needs_forward(self):
        class MinimalSeaiceAdr(SeaiceAdr):
            def __init__(self, q: int = 1):
                super().__init__(q)
                self.bias = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                return torch.stack((
                    x + self.bias,
                    y + self.bias,
                    x * 0 + self.bias,
                    y * 0 + self.bias,
                    x + y + self.bias,
                ), dim=-1)

        model = MinimalSeaiceAdr()
        predictions = model.predict(
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 2.0], dtype=np.float32),
            batch_size=1,
        )

        self.assertEqual(predictions['k'].shape, (2, 2))
        self.assertEqual(predictions['uhat_i'].shape, (2, 2, 1))


if __name__ == '__main__':
    unittest.main()
