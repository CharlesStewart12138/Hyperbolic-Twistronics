from __future__ import annotations

import sys
from pathlib import Path


EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))

from dyadic_ring import build_ring  # noqa: E402
from group_actions_v2 import enumerate_marked_group_v2  # noqa: E402


def test_exact_marked_image_orders_through_depth_six() -> None:
    observed = []
    for depth in range(1, 7):
        group = enumerate_marked_group_v2(
            "dyadic_ramified",
            build_ring(depth),
            with_auxiliary_p7=False,
            maximum_order=2_000_000,
        )
        observed.append(group.order)
    assert observed == [2, 2, 4, 8, 32, 32]
