"""Innovation, spillover and received spillover indices (Koschnick 2025, arXiv:2512.16587)."""
from __future__ import annotations

import re
import warnings
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from ._similarity import topk_indices, topk_mean, topk_values
from .embed import DEFAULT_MODEL, embed, resolve_device

MEASURES = ("innovation", "spillover", "received_spillover")


def field_slug(field) -> str:
    """Column-name suffix for a field: lower case, runs of other characters replaced by '_'."""
    return re.sub(r"[^0-9a-z]+", "_", str(field).lower()).strip("_")


class _Layout:
    """Embeddings reordered so that each field is a contiguous block sorted by year; pools become slices."""

    def __init__(self, E: torch.Tensor, years: np.ndarray, subj: np.ndarray):
        self.order = np.lexsort((years, subj))
        self.E = E[torch.as_tensor(self.order, device=E.device)]
        self.years = years[self.order]
        self.subj = subj[self.order]
        self.block = {}
        for f in pd.unique(self.subj):
            pos = np.flatnonzero(self.subj == f)
            self.block[f] = (pos[0], pos[-1] + 1)

    def window(self, f, lo: float, hi: float, lo_open: bool, hi_open: bool):
        """Slice (start, stop) of field f's titles with lo <(=) year <(=) hi."""
        a, b = self.block[f]
        y = self.years[a:b]
        s = a + np.searchsorted(y, lo, side="right" if lo_open else "left")
        e = a + np.searchsorted(y, hi, side="left" if hi_open else "right")
        return s, max(s, e)

    def rows(self, sl):
        return self.E[sl[0]:sl[1]]


def _ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    out = num / den
    out[den == 0] = float("nan")
    return out


def _check_fields(name, fields, available):
    if fields is None:
        return None
    fields = [str(fields)] if isinstance(fields, str) else [str(f) for f in fields]
    missing = [f for f in fields if f not in available]
    if missing:
        raise ValueError(f"{name}: {missing} not found in the subject column. Available fields: {sorted(available)}")
    return fields


def compute(
    df: pd.DataFrame,
    measure: Union[str, Sequence[str]] = "innovation",
    *,
    text: str = "text",
    year: str = "year",
    subject: str = "subject",
    tau: int = 20,
    k: int = 20,
    peers: Optional[int] = None,
    from_fields: Optional[Iterable] = None,
    to_fields: Optional[Iterable] = None,
    backward_window: Optional[int] = None,
    return_similarities: bool = False,
    complete_windows: bool = True,
    subset=None,
    embeddings: Optional[np.ndarray] = None,
    model=DEFAULT_MODEL,
    device: Optional[str] = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute text-based innovation and spillover indices for every row of ``df``.

    Parameters
    ----------
    df : DataFrame with one row per text (title, abstract, patent description, ...).
    measure : ``"innovation"``, ``"spillover"``, ``"received_spillover"``, or a list of these.
    text, year, subject : names of the columns holding the text, the (integer) publication year and the field.
    tau : length of the time windows in years (paper: 20).
    k : number of most similar texts averaged by the similarity function f^k (paper: 20).
    peers : number of own-field peers of the received spillover index (paper: 20; default ``k``).
    from_fields, to_fields : direction A -> B of the spillover measures. ``spillover`` is computed for texts in
        ``from_fields`` (default: all fields not in ``to_fields``) and requires ``to_fields``. ``received_spillover``
        is computed for texts in ``to_fields`` (default: all fields not in ``from_fields``) and requires
        ``from_fields``. With several fields on the other side, the index is the mean over fields and the
        per-field indices are returned as well.
    backward_window : innovation only. ``None`` (paper): the backward pool holds all earlier texts of the field;
        an integer w restricts it to the w preceding years.
    return_similarities : also return the components of each ratio (forward/backward similarities; similarity of
        the text and of its peers to the source field).
    complete_windows : leave the index missing (NaN) for texts whose time windows reach beyond the years covered
        by ``df``, where the pools would be truncated.
    subset : optional boolean mask (aligned with ``df``); the index is computed only for these rows, while all
        rows remain in the comparison pools.
    embeddings : optional precomputed embeddings, one row per row of ``df`` (e.g. from ``innovfromtext.embed``).
        If omitted, the texts are embedded with ``model``.
    model : Hugging Face name or local path of a sentence-transformers model (default: all-MiniLM-L6-v2; the paper
        uses SteamBERTh, ``model="juliuskoschnick/SteamBERTh"``).
    device : ``"auto"``/``None`` (GPU if available, else CPU), ``"gpu"``, ``"cpu"`` or a torch device string.
    batch_size, show_progress : passed to the encoder; ``show_progress`` also shows the progress of the indices.

    Returns
    -------
    A copy of ``df`` with the requested index columns appended (see the README for column names). Rows with a
    missing text, year or subject, rows outside ``subset`` and rows without the required pools get NaN.
    """
    measures = [measure] if isinstance(measure, str) else list(measure)
    bad = [m for m in measures if m not in MEASURES]
    if bad:
        raise ValueError(f"Unknown measure(s) {bad}; choose from {MEASURES}.")
    for c in (text, year, subject):
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not in df. Set text=, year= and subject= to your column names.")
    if tau < 1 or k < 1:
        raise ValueError("tau and k must be positive integers.")
    peers = k if peers is None else peers

    yr = pd.to_numeric(df[year], errors="coerce").to_numpy(float)
    txt = df[text]
    valid = ~np.isnan(yr) & df[subject].notna().to_numpy() & txt.notna().to_numpy()
    if subset is not None:
        subset = np.asarray(subset, dtype=bool)
        if subset.shape != (len(df),):
            raise ValueError("subset must be a boolean mask with one entry per row of df.")
    focal = valid & (subset if subset is not None else True)
    pos = np.flatnonzero(valid)
    if pos.size == 0:
        raise ValueError("No row has a text, a year and a subject.")
    years = np.floor(yr[pos]).astype(np.int64)
    subj = df[subject].to_numpy()[pos].astype(str)
    foc = focal[pos]
    available = set(subj)
    from_fields = _check_fields("from_fields", from_fields, available)
    to_fields = _check_fields("to_fields", to_fields, available)

    dev = resolve_device(device)
    if embeddings is None:
        E = embed(txt.iloc[pos].tolist(), model=model, device=str(dev), batch_size=batch_size,
                  show_progress=show_progress)
    else:
        E = np.asarray(embeddings, dtype=np.float32)
        if E.ndim != 2 or E.shape[0] != len(df):
            raise ValueError(f"embeddings must have one row per row of df ({len(df)}), got shape {E.shape}.")
        E = E[pos]
        norm = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.where(norm > 0, norm, 1)
    L = _Layout(torch.as_tensor(E, device=dev), years, subj)
    foc_s = foc[L.order]
    ymin, ymax = int(years.min()), int(years.max())
    res: dict = {}

    def groups(fields):
        """(field, year, slice of focal texts) for every field and year with texts to compute."""
        out = []
        for f in fields:
            a, b = L.block[f]
            for t0 in np.unique(L.years[a:b][foc_s[a:b]]):
                s, e = L.window(f, t0, t0, False, False)
                q = np.arange(s, e)[foc_s[s:e]]
                out.append((f, int(t0), q))
        return out

    def store(col, q, values):
        res.setdefault(col, np.full(len(L.years), np.nan))[q] = values.detach().cpu().numpy()

    def bar(it, desc):
        if not show_progress:
            return it
        from tqdm.auto import tqdm
        return tqdm(it, desc=desc)

    with torch.no_grad():
        if "innovation" in measures:
            for f, t0, q in bar(groups(sorted(available)), "innovation"):
                if complete_windows and (t0 + tau > ymax or (backward_window is not None and t0 - backward_window < ymin)):
                    continue
                lo = -np.inf if backward_window is None else t0 - backward_window
                back = L.window(f, lo, t0, False, True)
                forw = L.window(f, t0, t0 + tau, True, False)
                if back[0] == back[1] or forw[0] == forw[1]:
                    continue
                Q = L.E[torch.as_tensor(q, device=dev)]
                bs, fs = topk_mean(Q, L.rows(back), k), topk_mean(Q, L.rows(forw), k)
                store("innovation", q, _ratio(fs, bs))
                if return_similarities:
                    store("forward_similarity", q, fs)
                    store("backward_similarity", q, bs)

        if "spillover" in measures:
            if to_fields is None:
                raise ValueError("measure='spillover' needs to_fields, the field(s) B of the spillover A -> B.")
            src = from_fields or sorted(available - set(to_fields))
            for f, t0, q in bar(groups(src), "spillover"):
                if complete_windows and (t0 + tau > ymax or t0 - tau < ymin):
                    continue
                Q = L.E[torch.as_tensor(q, device=dev)]
                for B in to_fields:
                    if B == f:
                        continue
                    back = L.window(B, t0 - tau, t0, False, True)       # t0 - tau <= t <= t0 - 1
                    forw = L.window(B, t0, t0 + tau, True, False)       # t0 + 1 <= t <= t0 + tau
                    if back[0] == back[1] or forw[0] == forw[1]:
                        continue
                    bs, fs = topk_mean(Q, L.rows(back), k), topk_mean(Q, L.rows(forw), k)
                    store(f"spillover_to_{field_slug(B)}", q, _ratio(fs, bs))
                    if return_similarities:
                        store(f"forward_similarity_to_{field_slug(B)}", q, fs)
                        store(f"backward_similarity_to_{field_slug(B)}", q, bs)
            _aggregate(res, "spillover", [f"spillover_to_{field_slug(B)}" for B in to_fields], len(L.years))

        if "received_spillover" in measures:
            if from_fields is None:
                raise ValueError("measure='received_spillover' needs from_fields, the field(s) A of the spillover A -> B.")
            rec = to_fields or sorted(available - set(from_fields))
            for f, t0, q in bar(groups(rec), "received spillover"):
                if complete_windows and t0 - tau + 1 < ymin:
                    continue
                own = L.window(f, t0 - tau, t0, True, True)              # t0 - tau < t < t0, as in the paper's estimates
                if own[0] == own[1]:
                    continue
                Q = L.E[torch.as_tensor(q, device=dev)]
                O = L.rows(own)
                peer_pos = topk_indices(Q, O, peers)                       # peers of each text in its own field
                for A in from_fields:
                    if A == f:
                        continue
                    pool = L.window(A, t0 - tau, t0, True, True)
                    if pool[0] == pool[1]:
                        continue
                    P = L.rows(pool)
                    num = topk_mean(Q, P, k)
                    T = topk_values(O, P, k)                               # each own-field text's k closest links into A
                    flat = T[peer_pos].reshape(len(q), -1)                 # = top-k of the peers x pool matrix
                    den = flat.topk(min(k, flat.shape[1]), dim=1).values.mean(1)
                    store(f"received_spillover_from_{field_slug(A)}", q, _ratio(num, den))
                    if return_similarities:
                        store(f"title_similarity_from_{field_slug(A)}", q, num)
                        store(f"peer_similarity_from_{field_slug(A)}", q, den)
            _aggregate(res, "received_spillover", [f"received_spillover_from_{field_slug(A)}" for A in from_fields],
                       len(L.years))

    out = df.copy()
    inv = np.empty_like(L.order)
    inv[L.order] = np.arange(len(L.order))
    for col in _column_order(res, measures):
        full = np.full(len(df), np.nan)
        full[pos] = res[col][inv]
        out[col] = full
    return out


def _aggregate(res, name, cols, n):
    """Mean over source/target fields of the per-field ratios (ignoring missing ones); per-field columns are
    kept only when there is more than one field."""
    have = [c for c in cols if c in res]
    if not have:
        res[name] = np.full(n, np.nan)
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        res[name] = np.nanmean(np.column_stack([res[c] for c in have]), axis=1)
    if len(cols) == 1:
        res.pop(cols[0])


def _column_order(res, measures):
    order = {"innovation": ["innovation", "forward_similarity", "backward_similarity"]}
    cols = []
    for m in measures:
        if m == "innovation":
            cols += [c for c in order["innovation"] if c in res]
        elif m == "spillover":
            cols += ["spillover"] + sorted(c for c in res if c.startswith("spillover_to_"))
            cols += sorted(c for c in res if c.startswith(("forward_similarity_to_", "backward_similarity_to_")))
        else:
            cols += ["received_spillover"] + sorted(c for c in res if c.startswith("received_spillover_from_"))
            cols += sorted(c for c in res if c.startswith(("title_similarity_from_", "peer_similarity_from_")))
    return [c for c in dict.fromkeys(cols) if c in res]


def innovation(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Innovation index: forward similarity / backward similarity within the text's own field.
    Shortcut for ``compute(df, "innovation", ...)``."""
    return compute(df, "innovation", **kwargs)


def spillover(df: pd.DataFrame, to_fields, **kwargs) -> pd.DataFrame:
    """Spillover index A -> B of each text in A: forward / backward similarity to field B.
    Shortcut for ``compute(df, "spillover", to_fields=..., ...)``."""
    return compute(df, "spillover", to_fields=to_fields, **kwargs)


def received_spillover(df: pd.DataFrame, from_fields, **kwargs) -> pd.DataFrame:
    """Received spillover index A -> B of each text in B: its similarity to the recent past of A relative to that
    of its closest own-field peers. Shortcut for ``compute(df, "received_spillover", from_fields=..., ...)``."""
    return compute(df, "received_spillover", from_fields=from_fields, **kwargs)
