from typing import Optional, Tuple
import torch
from torch import nn


def build_net_for_demand() -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, Optional[nn.Module], Optional[nn.Module]]:
    treatment_1st_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 8))

    treatment_2nd_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(), nn.Linear(16, 8))

    treatment_proxy_net = nn.Sequential(nn.Linear(2, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(), nn.Linear(16, 8))

    outcome_proxy_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(), nn.Linear(16, 8))

    return treatment_1st_net, treatment_2nd_net, treatment_proxy_net, outcome_proxy_net, None, None
