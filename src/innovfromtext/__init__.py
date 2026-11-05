"""innovfromtext: text-based innovation and knowledge-spillover indices.

Implements the innovation, spillover and received spillover indices of Koschnick (2025), "Did a feedback mechanism
between propositional and prescriptive knowledge create modern growth?", arXiv:2512.16587.
"""
from .datasets import load_example
from .embed import DEFAULT_MODEL, embed, load_model, resolve_device
from .measures import MEASURES, compute, field_slug, innovation, received_spillover, spillover

__version__ = "0.2.0"

__all__ = [
    "compute", "innovation", "spillover", "received_spillover", "embed", "load_model", "resolve_device",
    "load_example", "field_slug", "DEFAULT_MODEL", "MEASURES", "__version__",
]
