import numpy as np
import torch
from typing import List
from causalflows.flows import CausalMAF, CausalNAF, CausalNCSF, CausalNSF, CausalUNAF
from zuko.flows import Flow


def hidden_conditional_adjacency(
    adjacency: torch.BoolTensor,
    num_hidden: int,
    conditional_vars: int,
) -> torch.BoolTensor:
    """Return the effective conditional-to-hidden relation used by conditional modules."""
    assert adjacency is not None, "adjacency must be provided"
    assert conditional_vars > 0, "conditional_vars must be greater than 0"

    adjacency_cz = adjacency[:num_hidden, -conditional_vars:]
    adjacency_zc = adjacency_cz.T
    return adjacency[-conditional_vars:, :num_hidden] | adjacency_zc


def has_conditional_prior(
    adjacency: torch.BoolTensor,
    num_hidden: int,
    conditional_vars: int,
) -> bool:
    """Whether the latent prior should depend on the conditional variables."""
    return bool(hidden_conditional_adjacency(adjacency, num_hidden, conditional_vars).any().item())


class PriorConditioner(Flow):
    r"""PriorConditioner class that instantiates a specific causal flow based on the flow type.
    It is defined as a flow, but transforms and bases come from causal flows, for simplicity.

    RETURNS P_{Z|X}(z|x)
    """

    _flow_classes = {
        'maf': CausalMAF,
        'nsf': CausalNSF,
        'naf': CausalNAF,
        'ncsf': CausalNCSF,
        'unaf': CausalUNAF,
    }

    def __init__(self, flow_type: str, num_hidden: int, conditional_vars: int, adjacency: torch.BoolTensor = None,
                 hidden_indices: List[int] = None, *args, **kwargs):

        assert flow_type in self._flow_classes, f"Unknown flow type: {flow_type}. Choose from {list(self._flow_classes)}"
        assert num_hidden > 0, "num_hidden must be greater than 0. Otherwise, use a Causal Flow directly."
        assert conditional_vars > 0, ("conditional_vars must be greater than 0. Otherwise, use an isotropic normal prior "
                                      "directly.")

        if adjacency is not None:
            adjacency_copy = adjacency.clone()
            adjacency_copy[-conditional_vars:, :num_hidden] = hidden_conditional_adjacency(
                adjacency_copy,
                num_hidden,
                conditional_vars,
            )
            adjacency = adjacency_copy

            if hidden_indices is None:
                hidden_indices = list(range(num_hidden))

            conditional_indices = list(range(adjacency.shape[0] - conditional_vars, adjacency.shape[0]))
            keep_indices = hidden_indices + conditional_indices

            adjacency_cz = adjacency[hidden_indices][:, conditional_indices]
            adjacency_zc = adjacency_cz.T
            adjacency[conditional_indices][:, hidden_indices] |= adjacency_zc

            adjacency = adjacency[keep_indices][:, keep_indices]
            remapped_hidden_indices = list(range(len(hidden_indices)))
        else:
            remapped_hidden_indices = None

        if adjacency is not None:
            if remapped_hidden_indices is not None:
                # Hidden variables could not be roots (not included in the theory, but useful)
                # If that is the case, hidden_indices must be provided
                # e.g. hidden_indices = [1,3], x1->z1, x2->z3
                assert len(remapped_hidden_indices) == num_hidden, "Length of hidden indices must match num_hidden"
                for k in remapped_hidden_indices:
                    parents_of_z_k = torch.where(adjacency[k, :k] == 1)[0]
                    mask = ~torch.isin(parents_of_z_k, torch.tensor(remapped_hidden_indices))
                    parents_of_z_k = parents_of_z_k[mask]
                    adjacency[parents_of_z_k, k] = 1
                    # account for collider association
                    children_z = torch.where(adjacency[:, k] == 1)[0]
                    for j in children_z:
                        other_parents = torch.where(adjacency[j, k + 1:j] == 1)[0] + k + 1
                        adjacency[other_parents, k] = 1

                adjacency_ez = adjacency[np.ix_(remapped_hidden_indices, remapped_hidden_indices)]  # only hidden variables
                # for adjacency x->z, remove the rows and columns of the hidden variables
                mask = torch.zeros(adjacency.shape[0], dtype=torch.bool)
                mask[remapped_hidden_indices] = True
                adjacency_xz = adjacency[~mask][:, mask]
            else:
                # account for collider association
                adjacency_copy = adjacency.clone()
                for k in range(num_hidden):
                    children_z = torch.where(adjacency_copy[:, k] == 1)[0]
                    for j in children_z:
                        other_parents = torch.where(adjacency_copy[j, k + 1:j] == 1)[0] + k + 1
                        adjacency_copy[other_parents, k] = 1

                adjacency_ez = adjacency_copy[:num_hidden, :num_hidden]
                adjacency_xz = adjacency_copy[num_hidden:, :num_hidden]

            adjacency = torch.cat((adjacency_ez, adjacency_xz.T), dim=-1)

        flow_class = self._flow_classes[flow_type]
        flow = flow_class(adjacency=adjacency, *args, **kwargs)

        self.latent_dim = num_hidden
        self.adjacency = adjacency

        # Initialize as a CausalFlow with the selected flow's transform and base
        super().__init__(flow.transform, flow.base)
