# innovfromtext

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/juliuskoschnick/innovfromtext/blob/main/examples/innovfromtext_example.ipynb)
[![tests](https://github.com/juliuskoschnick/innovfromtext/actions/workflows/tests.yml/badge.svg)](https://github.com/juliuskoschnick/innovfromtext/actions/workflows/tests.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Model: SteamBERTh](https://img.shields.io/badge/%F0%9F%A4%97%20model-SteamBERTh-yellow)](https://huggingface.co/juliuskoschnick/SteamBERTh)

Text-based measures of innovation and knowledge spillovers. `innovfromtext` computes the **innovation index**, the
**spillover index** and the **received spillover index** of
[Koschnick (2025)](https://arxiv.org/abs/2512.16587) for any collection of dated texts that are assigned to fields:
book titles, abstracts, patent descriptions, articles. The measures need no citation data, only the texts, their
years and their fields.

In [Koschnick (2025)](https://arxiv.org/abs/2512.16587), the measure uses
[SteamBERTh](https://huggingface.co/juliuskoschnick/SteamBERTh) embeddings, a historical BERT model fine-tuned on
seventeenth- and eighteenth-century scientific and technical texts. For other applications, any other
[sentence-transformers](https://www.sbert.net) model can be used instead. As a default for modern data, the package
calls [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Installation

```bash
pip install git+https://github.com/juliuskoschnick/innovfromtext.git
```

Requires Python 3.9 or later. A GPU is used automatically when PyTorch finds one. However, everything also runs on a CPU.

## Quick start

```python
import innovfromtext as ift

df = ift.load_example()     # Philosophical Transactions titles 1665-1800 and English patents 1700-1800

out = ift.compute(df, "innovation", model="juliuskoschnick/SteamBERTh")      # historical texts: SteamBERTh, modern text: choose your own model or use all-MiniLM-L6-v2 as default

out = ift.compute(df, "received_spillover",
                  from_fields=["Philosophical Transactions"], to_fields=["Patents"], model="juliuskoschnick/SteamBERTh")
```

The data need one row per text with the text, its year and its field; name your columns with `text=`, `year=` and
`subject=`.

`compute` returns `df` with the index appended as a new column. The
[example notebook](examples/innovfromtext_example.ipynb) walks through all three measures and runs on Google Colab
without installing anything locally.
A second, short [demo on film plots](examples/innovfromtext_movies.ipynb) illustrates the functioning of the
innovation index by applying it to the plots of 35,000 American films and calculating which ones are the most
innovative.

## Intuition

A text is innovative if it is close to the future of its field but distant from its past: it is novel (unlike what
came before) and impactful (like what came after).

$$\text{Innovation}_i = \frac{\text{similarity of } i \text{ to the future of its field}}{\text{similarity of } i \text{ to the past of its field}}$$

The same logic applies across fields: a text $i$ in field $A$ creates a spillover into field $B$ if it is close to
the future of field $B$ but distant from the past of field $B$.

## The measures

All three measures compare texts through the similarity function

$$f^k(v, \mathcal{P}) = \text{mean of the } k \text{ largest cosine similarities between text } v \text{ and the texts of pool } \mathcal{P},$$

computed in the embedding space of the language model. Let text $i$ of field $B$ be published in year $t_0$, and
let $\tau$ be the window length (default $\tau = k = 20$). Texts published in the same year as $t_0$ belong to no
pool.

| Measure | Computed for | Numerator | Denominator |
|---|---|---|---|
| `innovation` | every text | $f^k$ to its own field's texts of $t_0+1, \dots, t_0+\tau$ (impact) | $f^k$ to all earlier texts of its own field (novelty) |
| `spillover` $A \to B$ | texts of $A$ (`from_fields`) | $f^k$ to the texts of $B$ of $t_0+1, \dots, t_0+\tau$ | $f^k$ to the texts of $B$ of $t_0-\tau, \dots, t_0-1$ |
| `received_spillover` $A \to B$ | texts of $B$ (`to_fields`) | $f^k$ to the texts of $A$ of $t_0-\tau+1, \dots, t_0-1$ | the same for the text's peers: the mean of the $k$ largest similarities between its `peers` most similar texts of $B$ (same years) and the texts of $A$ |

A text is innovative if its field's future resembles it more than its field's past did (index above one). A text
of $A$ creates a spillover into $B$ if it resembles $B$'s future more than $B$'s past. A text of $B$ received a
spillover from $A$ if it came closer to the recent past of $A$ than its closest predecessors in $B$ had been. With
several fields on the other side ($B$ for `spillover`, $A$ for `received_spillover`), the index is the mean of the
per-field indices, as in the paper, and the per-field indices are returned as well.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `measure` | `"innovation"` | `"innovation"`, `"spillover"`, `"received_spillover"`, or a list of these |
| `text`, `year`, `subject` | `"text"`, `"year"`, `"subject"` | column names of the text, the integer publication year and the field |
| `tau` | 20 | length of the time windows in years |
| `k` | 20 | number of most similar texts averaged by $f^k$ |
| `peers` | `k` | number of own-field peers of the received spillover index |
| `from_fields`, `to_fields` | | direction $A \to B$ of the spillover measures; see the table above |
| `backward_window` | `None` | innovation only: `None` compares with all earlier texts, an integer *w* with the *w* preceding years |
| `return_similarities` | `False` | also return the numerator and denominator of each ratio |
| `complete_windows` | `True` | missing index for texts whose windows reach beyond the years covered by the data |
| `subset` | `None` | boolean mask: compute the index only for these rows (all rows stay in the pools) |
| `embeddings` | `None` | precomputed embeddings, one row per row of `df` |
| `model` | `"sentence-transformers/all-MiniLM-L6-v2"` | any sentence-transformers model name or local path; SteamBERTh: `"juliuskoschnick/SteamBERTh"` |
| `device` | `"auto"` | `"auto"` (GPU if available, else CPU), `"gpu"`, `"cpu"`, or a torch device such as `"cuda:1"` |
| `batch_size`, `show_progress` | 32, `True` | encoding batch size; progress bars |

`ift.innovation(df, ...)`, `ift.spillover(df, to_fields, ...)` and `ift.received_spillover(df, from_fields, ...)` are
shortcuts for `compute` with the measure set.

## Output columns

| Measure | Columns |
|---|---|
| innovation | `innovation`; with `return_similarities`: `forward_similarity`, `backward_similarity` |
| spillover | `spillover`; with several target fields also `spillover_to_<field>`; with `return_similarities`: `forward_similarity_to_<field>`, `backward_similarity_to_<field>` |
| received spillover | `received_spillover`; with several source fields also `received_spillover_from_<field>`; with `return_similarities`: `title_similarity_from_<field>` (numerator), `peer_similarity_from_<field>` (denominator) |

`<field>` is the field name in lower case with spaces and other characters replaced by `_`
(`"Applied physics"` becomes `applied_physics`; see `ift.field_slug`). Rows with a missing text, year or subject,
and rows without the required pools, get a missing value.

## Practical notes

- **Embed once.** The bottleneck of calculating innovation and spillovers is the calculation of the embeddings.
  Hence, if you're calculating multiple measures, only embed once. To compute several measures or variants, use
  `emb = ift.embed(df["title"])` and pass `embeddings=emb`, or pass a list of measures to `compute`.
- **GPU or CPU.** Embedding text is significantly faster on a GPU than a CPU (seconds vs. minutes for a few
  thousand texts). All embeddings are held on the device during the computation: 300,000 texts need about 1 GB.
- **Start and end of the data.** Texts in the last $\tau$ years have incomplete forward windows and get a missing
  index (unless `complete_windows=False`). Texts in the first years of the data have few earlier texts to compare
  with, so their indices are noisy.
- **Models.** The default, all-MiniLM-L6-v2, is a small model for modern texts. For historical texts, use SteamBERTh
  as in the paper: `model="juliuskoschnick/SteamBERTh"`.
- **Exporting.** `out.to_csv("indices.csv", index=False)` gives a file for Stata or R. Keep an identifier column in
  `df` to merge the indices back.

## Citation

If you use the package or the measures, please cite:

```bibtex
@misc{koschnick2025feedback,
  title         = {Did a feedback mechanism between propositional and prescriptive knowledge create modern growth?},
  author        = {Koschnick, Julius},
  year          = {2025},
  eprint        = {2512.16587},
  archivePrefix = {arXiv},
  primaryClass  = {econ.GN},
  doi           = {10.48550/arXiv.2512.16587},
  url           = {https://arxiv.org/abs/2512.16587}
}
```

If you use SteamBERTh, please also cite MacBERTh, its base model
([Manjavacas and Fonteyn 2022](https://doi.org/10.46298/jdmdh.9152);
[Manjavacas Arevalo and Fonteyn 2021](https://aclanthology.org/2021.nlp4dh-1.4/)); see the
[model card](https://huggingface.co/juliuskoschnick/SteamBERTh).

## License

The package is licensed under the [GNU General Public License v3.0 or later](LICENSE). SteamBERTh is licensed under
MIT, as its base model MacBERTh. The example data are the article titles of the *Philosophical Transactions* of the
Royal Society, 1665–1800 (from JSTOR), and the short descriptions of English patents, 1700–1800 (Woodcroft 1854).
