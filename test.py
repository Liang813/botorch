import torch
from botorch.qmc.normal import NormalQMCEngine
normal_qmc_engine = NormalQMCEngine(d=2, seed=343, inv_transform=True)
samples = normal_qmc_engine.draw(10000, dtype=torch.float)
print(samples.max())
