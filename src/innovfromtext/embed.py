"""Device selection and text embeddings."""
from __future__ import annotations

from typing import Iterable, Optional, Union

import numpy as np
import torch

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # SteamBERTh (the paper): "juliuskoschnick/SteamBERTh"

_MODELS: dict = {}


def resolve_device(device: Optional[str] = None) -> torch.device:
    """Pick the device for all computations.

    ``None`` or ``"auto"`` uses the first CUDA GPU if one is available and the CPU otherwise. ``"gpu"`` asks for a GPU
    (CUDA, or Apple ``"mps"`` if there is no CUDA) and fails if there is none. Any torch device string
    (``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"``) is passed on.
    """
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    d = str(device).lower()
    if d == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("device='gpu' was requested, but torch finds no GPU. Use device='cpu' or 'auto'.")
    if d.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device='{device}' was requested, but torch finds no CUDA GPU. Use device='cpu' or 'auto'.")
    return torch.device(d)


def load_model(model: Union[str, "SentenceTransformer"] = DEFAULT_MODEL, device: Optional[str] = None):
    """Load a sentence-transformers model by Hugging Face name or local path (cached per name and device)."""
    from sentence_transformers import SentenceTransformer

    if isinstance(model, SentenceTransformer):
        return model
    dev = resolve_device(device)
    key = (str(model), str(dev))
    if key not in _MODELS:
        _MODELS[key] = SentenceTransformer(str(model), device=str(dev))
    return _MODELS[key]


def embed(
    texts: Iterable[str],
    model: Union[str, "SentenceTransformer"] = DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed texts with a sentence-transformers model (default: all-MiniLM-L6-v2).

    Returns a float32 array of shape (number of texts, embedding dimension) with unit-length rows, so that dot
    products are cosine similarities. Embed once and pass the result to ``compute(..., embeddings=...)`` to
    compute several measures without encoding the texts again.
    """
    texts = ["" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t) for t in texts]
    m = load_model(model, device)
    E = m.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=show_progress)
    return E.astype(np.float32)
