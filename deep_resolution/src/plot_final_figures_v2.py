from __future__ import annotations

from matplotlib.text import Text

import plot_final_figures as implementation


_set_text = Text.set_text


def _sanitized_set_text(self: Text, value) -> None:
    if isinstance(value, str):
        value = value.replace("\rho", r"\rho").replace("\to", r"\to")
    _set_text(self, value)


Text.set_text = _sanitized_set_text


if __name__ == "__main__":
    raise SystemExit(implementation.main())

