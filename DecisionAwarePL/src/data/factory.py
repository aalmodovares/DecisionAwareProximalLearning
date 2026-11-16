from __future__ import annotations

from src.config import DataConfig
from src.data.simulator import Wu2024Simulator
from src.data.tcga_semisynthetic import SEMISYNTHETIC_TCGA_VERSION, SemisyntheticTCGASimulator


def make_simulator(config: DataConfig):
    if config.version == SEMISYNTHETIC_TCGA_VERSION:
        return SemisyntheticTCGASimulator(config)
    return Wu2024Simulator(config)
