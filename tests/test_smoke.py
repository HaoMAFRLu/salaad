import torch

from salaad.salad_solver import SALAD


def test_salad_solver_updates_small_matrix():
    x = torch.randn(8, 6)
    params = {
        "energy": 0.9,
        "init_energy": 0.0,
        "is_init": False,
        "device": "cpu",
        "rate_rank": 0.5,
        "rate_sparsity": 0.1,
        "rho_dict": {
            "rho": 1e-3,
            "mode": "fixed",
            "start_epoch": 2,
            "coeff_rho": 0.1,
            "coeff_rho_min": 0.01,
            "coeff_rho_max": 1500.0,
            "rho_rate": 1.0,
        },
        "alpha_dict": {
            "init": 1e-6,
            "mode": "adaptive",
            "rate_decay": 0.02,
        },
        "beta_dict": {
            "init": 1e-6,
            "mode": "adaptive",
            "rate_decay": 0.02,
        },
    }

    solver = SALAD("test_layer", params, x, nr_layers=1, is_full=True)
    solver.update_L()
    solver.update_S()
    solver.update_Y()

    assert solver.L.shape == x.shape
    assert solver.S.shape == x.shape
    assert solver.Y.shape == x.shape
