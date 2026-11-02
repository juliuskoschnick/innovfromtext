"""Chunked top-k cosine similarities on unit-normalised embeddings."""
from __future__ import annotations

import torch

QUERY_CHUNK = 2048
POOL_CHUNK = 8192


def _topk(Q: torch.Tensor, P: torch.Tensor, k: int, with_indices: bool = False):
    """The k largest dot products between each row of Q and the rows of P.

    Never materialises the full |Q| x |P| matrix. Returns values (and positions in P) of shape (|Q|, min(k, |P|)).
    """
    n, m = Q.shape[0], P.shape[0]
    k = min(k, m)
    vals_out = torch.empty((n, k), dtype=torch.float32, device=Q.device)
    idx_out = torch.empty((n, k), dtype=torch.long, device=Q.device) if with_indices else None
    for qs in range(0, n, QUERY_CHUNK):
        q = Q[qs:qs + QUERY_CHUNK]
        best_v = best_i = None
        for ps in range(0, m, POOL_CHUNK):
            s = (q @ P[ps:ps + POOL_CHUNK].T).float()
            if with_indices:
                i = torch.arange(ps, ps + s.shape[1], device=Q.device).expand_as(s)
            if best_v is not None:
                s = torch.cat([best_v, s], 1)
                if with_indices:
                    i = torch.cat([best_i, i], 1)
            top = s.topk(min(k, s.shape[1]), dim=1)
            best_v = top.values
            if with_indices:
                best_i = i.gather(1, top.indices)
        vals_out[qs:qs + QUERY_CHUNK] = best_v
        if with_indices:
            idx_out[qs:qs + QUERY_CHUNK] = best_i
    return (vals_out, idx_out) if with_indices else vals_out


def topk_mean(Q: torch.Tensor, P: torch.Tensor, k: int) -> torch.Tensor:
    """f^k: mean of the k largest cosine similarities between each row of Q and the pool P (NaN for an empty pool)."""
    if P.shape[0] == 0:
        return torch.full((Q.shape[0],), float("nan"), device=Q.device)
    return _topk(Q, P, k).mean(1)


def topk_values(Q: torch.Tensor, P: torch.Tensor, k: int) -> torch.Tensor:
    """The k largest cosine similarities between each row of Q and the pool P."""
    return _topk(Q, P, k)


def topk_indices(Q: torch.Tensor, P: torch.Tensor, k: int) -> torch.Tensor:
    """Positions in P of the k rows most similar to each row of Q."""
    return _topk(Q, P, k, with_indices=True)[1]
