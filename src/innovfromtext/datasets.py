"""Example data."""
from __future__ import annotations

from importlib.resources import files

import pandas as pd


def load_example() -> pd.DataFrame:
    """7,581 texts on science and technology in England: the article titles of the *Philosophical Transactions* of
    the Royal Society, 1665-1800 (5,554, from JSTOR; Latin titles translated into English), and the short
    descriptions of English patents, 1700-1800 (2,027, from Woodcroft 1854).

    Columns: ``id`` (DOI or JSTOR link of the article; patent number), ``year``, ``subject``
    (``"Philosophical Transactions"`` or ``"Patents"``) and ``text``.
    """
    with files("innovfromtext").joinpath("data/example.csv").open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh)
