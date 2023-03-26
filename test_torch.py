import time

import numpy as np
import torch

from oscillatoryInterference import GridCell
from oscillatoryInterference_torch import GridCell as GridCellTorch
from pathTools import generateQ1Test
from see import firing_voronoi

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)

    path = generateQ1Test(
        speed=20.0,
        samples_per_second=50.0,
        arena_size=40
    )

    model_params = {
        "samples_per_second": 50.0,
        "theta_frequency": 9.0,
        "somatic_phase_offset": 0.0,
        "cm_per_cycle": 10.0,
        "n_dendritic": 6,
        "orientation": 0.0,
    }

    # Numpy version
    gc_np = GridCell(**model_params, offset_proportion=1.0)
    start = time.time()
    gc_np.test(path)

    # extract positions and firing rates
    _, positions_np, firing_np = zip(*gc_np.firing_history)
    positions_np = np.array(positions_np)
    firing_np = np.array(firing_np)

    # print(f"gc_np elapsed time: {time.time() - start:.5f} seconds")
    # print(f"numpy firing: {firing_np.shape}")
    # print(f"numpy positions: {positions_np.shape}")
    # print()

    # Torch version
    gc_torch = GridCellTorch(**model_params, offset_proportions=(0.7, 0.2))
    start = time.time()
    with torch.no_grad():
        firing_torch = gc_torch.batch_record(path)
    positions_torch = gc_torch.positions

    firing_voronoi(positions_torch.numpy(), firing_torch.numpy(), "torch")
    exit()

    print(f"gc_torch elapsed time: {time.time() - start:.5f} seconds")
    print(f"torch firing: {firing_torch.numpy().shape}")
    print(f"torch positions: {positions_torch.numpy().shape}")

    # compare firing rates between numpy and torch
    assert torch.allclose(torch.tensor(firing_np).double(), firing_torch.double()), \
        f"{(torch.tensor(firing_np).double() - firing_torch.double()).abs().max()} >= atol (1e-08) or rtol (1e-05)"
    assert torch.allclose(torch.tensor(positions_np).double(), positions_torch.double())

    # plot firing rates
    firing_voronoi(positions_np, firing_np, "numpy")
    firing_voronoi(positions_torch.numpy(), firing_torch.numpy(), "torch")
