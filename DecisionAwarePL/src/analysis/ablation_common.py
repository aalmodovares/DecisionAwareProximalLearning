from __future__ import annotations

from typing import Sequence

import pandas as pd


def ordered_methods(observed_methods: Sequence[str], method_order: Sequence[str] | None = None) -> list[str]:
    observed = [str(method) for method in observed_methods if pd.notna(method)]
    observed_set = set(observed)
    order = [str(method) for method in (method_order or []) if str(method) in observed_set]
    order.extend(sorted(method for method in observed if method not in set(order)))
    return order


