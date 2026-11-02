"""Checks the chunked implementation against a direct per-text implementation of the paper's definitions."""
import numpy as np
import pandas as pd
import pytest

import innovfromtext as ift
from innovfromtext import _similarity

FIELDS = ["Astronomy", "Navigation", "Trades"]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"text": [f"title {i}" for i in range(n)],
                       "year": rng.integers(1700, 1761, n),
                       "subject": rng.choice(FIELDS, n, p=[0.3, 0.3, 0.4])})
    E = (rng.normal(size=(n, 16)) + 1.0).astype(np.float32)     # shared component: positive similarities, as for texts
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    return df, E


@pytest.fixture(params=[(2048, 8192), (7, 5)], ids=["default-chunks", "small-chunks"])
def chunks(request, monkeypatch):
    monkeypatch.setattr(_similarity, "QUERY_CHUNK", request.param[0])
    monkeypatch.setattr(_similarity, "POOL_CHUNK", request.param[1])


def fk(v, P, k):
    if len(P) == 0:
        return np.nan
    s = np.sort(P @ v)[::-1]
    return s[:k].mean()


def ref_innovation(df, E, tau, k):
    y, f = df.year.to_numpy(), df.subject.to_numpy()
    out = np.full((len(df), 3), np.nan)
    for i in range(len(df)):
        back = E[(f == f[i]) & (y < y[i])]
        forw = E[(f == f[i]) & (y > y[i]) & (y <= y[i] + tau)]
        if len(back) and len(forw):
            bs, fs = fk(E[i], back, k), fk(E[i], forw, k)
            out[i] = fs / bs, fs, bs
    return out


def ref_spillover(df, E, B, tau, k):
    y, f = df.year.to_numpy(), df.subject.to_numpy()
    out = np.full(len(df), np.nan)
    for i in range(len(df)):
        if f[i] == B:
            continue
        back = E[(f == B) & (y >= y[i] - tau) & (y < y[i])]
        forw = E[(f == B) & (y > y[i]) & (y <= y[i] + tau)]
        if len(back) and len(forw):
            out[i] = fk(E[i], forw, k) / fk(E[i], back, k)
    return out


def ref_received(df, E, A, tau, k, rho):
    y, f = df.year.to_numpy(), df.subject.to_numpy()
    out = np.full((len(df), 3), np.nan)
    for i in range(len(df)):
        if f[i] == A:
            continue
        win = (y > y[i] - tau) & (y < y[i])
        own, pool = E[(f == f[i]) & win], E[(f == A) & win]
        if len(own) == 0 or len(pool) == 0:
            continue
        peers = own[np.argsort(-(own @ E[i]))[:rho]]
        num = fk(E[i], pool, k)
        den = np.sort((peers @ pool.T).ravel())[::-1][:k].mean()
        out[i] = num / den, num, den
    return out


def test_innovation(data, chunks):
    df, E = data
    out = ift.compute(df, "innovation", tau=10, k=5, embeddings=E, return_similarities=True,
                      complete_windows=False, device="cpu", show_progress=False)
    ref = ref_innovation(df, E, 10, 5)
    np.testing.assert_allclose(out[["innovation", "forward_similarity", "backward_similarity"]].to_numpy(), ref,
                               rtol=1e-5, atol=1e-6)
    assert np.isfinite(ref[:, 0]).sum() > 400


def test_spillover(data, chunks):
    df, E = data
    out = ift.compute(df, "spillover", to_fields=["Astronomy", "Navigation"], tau=10, k=5, embeddings=E,
                      complete_windows=False, device="cpu", show_progress=False)
    a, n = ref_spillover(df, E, "Astronomy", 10, 5), ref_spillover(df, E, "Navigation", 10, 5)
    trades = (df.subject == "Trades").to_numpy()
    np.testing.assert_allclose(out.spillover_to_astronomy[trades], a[trades], rtol=1e-5)
    np.testing.assert_allclose(out.spillover_to_navigation[trades], n[trades], rtol=1e-5)
    np.testing.assert_allclose(out.spillover[trades], np.nanmean(np.c_[a, n], 1)[trades], rtol=1e-5)
    assert out.spillover[~trades].isna().all()            # default from_fields: fields not in to_fields


def test_received_spillover(data, chunks):
    df, E = data
    out = ift.compute(df, "received_spillover", from_fields=["Astronomy", "Navigation"], tau=10, k=5, peers=4,
                      embeddings=E, return_similarities=True, complete_windows=False, device="cpu",
                      show_progress=False)
    a, n = ref_received(df, E, "Astronomy", 10, 5, 4), ref_received(df, E, "Navigation", 10, 5, 4)
    trades = (df.subject == "Trades").to_numpy()
    np.testing.assert_allclose(out.received_spillover_from_astronomy[trades], a[trades, 0], rtol=1e-5)
    np.testing.assert_allclose(out.title_similarity_from_navigation[trades], n[trades, 1], rtol=1e-5)
    np.testing.assert_allclose(out.peer_similarity_from_navigation[trades], n[trades, 2], rtol=1e-5)
    np.testing.assert_allclose(out.received_spillover[trades], np.nanmean(np.c_[a[:, 0], n[:, 0]], 1)[trades],
                               rtol=1e-5)


def test_single_field_and_k_larger_than_pools(data):
    df, E = data
    out = ift.received_spillover(df, from_fields="Astronomy", to_fields=["Trades"], tau=10, k=50, peers=30,
                                 embeddings=E, complete_windows=False, device="cpu", show_progress=False)
    ref = ref_received(df, E, "Astronomy", 10, 50, 30)[:, 0]
    trades = (df.subject == "Trades").to_numpy()
    assert "received_spillover_from_astronomy" not in out           # one field: only the index itself
    np.testing.assert_allclose(out.received_spillover[trades], ref[trades], rtol=1e-5)
    assert out.received_spillover[df.subject == "Navigation"].isna().all()


def test_complete_windows_subset_and_missing(data):
    df, E = data
    df = df.copy()
    df.loc[0, "text"] = None
    sub = (df.year % 2 == 0).to_numpy()
    out = ift.compute(df, ["innovation", "received_spillover"], from_fields=["Astronomy"], tau=10, k=5,
                      embeddings=E, subset=sub, device="cpu", show_progress=False)
    assert len(out) == len(df) and list(out.columns[:3]) == ["text", "year", "subject"]
    assert np.isnan(out.innovation[0])
    assert out.innovation[~sub].isna().all() and out.received_spillover[~sub].isna().all()
    assert out.innovation[df.year > 1750].isna().all()             # forward window beyond 1760
    assert out.received_spillover[df.year < 1709].isna().all()     # backward window before 1700
    assert out.innovation[sub & (df.year <= 1750) & (df.year > 1700)].notna().mean() > 0.9


def test_errors(data):
    df, E = data
    with pytest.raises(ValueError, match="to_fields"):
        ift.compute(df, "spillover", embeddings=E, device="cpu", show_progress=False)
    with pytest.raises(ValueError, match="not found"):
        ift.compute(df, "received_spillover", from_fields=["Physics"], embeddings=E, device="cpu")
    with pytest.raises(ValueError, match="one row per row"):
        ift.compute(df, embeddings=E[:10], device="cpu")
    with pytest.raises(KeyError, match="title"):
        ift.compute(df, text="title", embeddings=E, device="cpu")
    with pytest.raises(ValueError, match="Unknown measure"):
        ift.compute(df, "novelty", embeddings=E, device="cpu")


def test_device():
    assert ift.resolve_device("cpu").type == "cpu"
    assert ift.resolve_device(None).type in ("cpu", "cuda")


def test_example_data():
    df = ift.load_example()
    assert list(df.columns) == ["id", "year", "subject", "text"]
    assert len(df) == 7581 and set(df.subject) == {"Philosophical Transactions", "Patents"}
    assert df.id.is_unique and df.text.notna().all()
