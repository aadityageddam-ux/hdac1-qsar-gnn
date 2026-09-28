# What the split changes, and what it doesn't: fingerprint and graph models for HDAC1

**Does a graph neural network learn anything about HDAC1 inhibition that a 2048-bit ECFP4 random
forest does not — and how much does that answer depend on how the test set was chosen?**

**Result: no, and the split changes how good every model looks without changing which one wins.**
On a leak-free Murcko scaffold split of 6,816 ChEMBL compounds, three models — a tuned random
forest, a hand-rolled GINE network, and Chemprop's D-MPNN — land within 0.006 pIC50 log units of
each other, with **1 of 27 model-pair × metric comparisons reaching significance, which is what 27
tests at α = 0.05 produce by chance.** Re-run the identical models on the identical data under a
*random* split and every model improves by roughly 0.12–0.13 log units — about 17% lower RMSE, for
free, from the evaluation protocol alone — while the forest and the D-MPNN remain tied.

So the quantity worth reporting is not which architecture wins. It is that **a random-split
evaluation of this dataset reports ~17% better RMSE than an honest one**, on identical data and
identical models.

> **A correction, kept in place because it is the most useful thing in this repository.** An
> earlier version of this README claimed something stronger: that a random split flatters the
> fingerprint model *more than twice as much* as a graph model, enough to reverse the apparent
> winner. That was measured against my hand-rolled GINE network, and it did not survive adding
> Chemprop. The D-MPNN's optimism (+0.1246) is essentially identical to the forest's (+0.1285),
> and under a random split the two are still tied. The apparent asymmetry was a property of my
> weaker model, not of graph models. The full numbers are below and the episode is written up in
> `AI_USAGE.md`.

## The two results

### 1. Three models, no significant difference

Test fold, 1,022 compounds. Every model is scored by the same code (`src/evaluate.py`), on the
same split (verified by hash), with the same 2,000 bootstrap resamples. All three were tuned or
early-stopped on the validation fold only, refit on train + val, and run over five seeds.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Spearman ↑ | ROC-AUC ↑ | PR-AUC ↑ | MCC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean predictor | 1.1060 | 0.9007 | −0.017 | — | 0.500 | 0.309 | 0.000 |
| Majority class | 1.2015 | 0.9499 | −0.200 | — | 0.500 | 0.309 | 0.000 |
| Random scores | 2.2340 | 1.8290 | −3.147 | 0.009 | 0.483 | 0.293 | 0.011 |
| 1-NN Tanimoto | 0.9440 | 0.6530 | 0.259 | 0.657 | 0.818 | 0.659 | 0.466 |
| Random forest (ECFP4 + 12 descriptors) | 0.7367 | 0.5640 | 0.549 | 0.7594 | 0.8779 | 0.7784 | 0.5636 |
| GINE graph neural network | 0.7357 | 0.5535 | 0.550 | 0.7456 | 0.8670 | 0.7563 | 0.5250 |
| **Chemprop D-MPNN** (field standard) | **0.7312** | **0.5407** | **0.556** | **0.7644** | **0.8801** | **0.7812** | **0.5650** |

Headline comparison — random forest against Chemprop, the model the field actually benchmarks
against: **Δ RMSE +0.0054, 95% CI [−0.0280, +0.0372], p = 0.757.**

Chemprop is numerically best on every metric, and none of it survives a significance test. Across
all three model pairs and all nine metrics, exactly one comparison reached p < 0.05 (random forest
vs Chemprop on balanced accuracy, p = 0.030). Twenty-seven tests at α = 0.05 are expected to throw
up about 1.4 false positives, so that result is reported and not believed.

**The Δ row is the headline, not the marginal intervals.** Two overlapping confidence intervals do
not establish that two models are indistinguishable. The paired bootstrap scores every model on the
same resamples and takes the difference, which is the correct test — and here it confirms the tie
rather than merely failing to reject it.

### 2. A random split inflates every model by about the same amount

The same models, the same data, the same fold sizes — differing only in how compounds were
assigned:

| Model | random split | scaffold split | optimism |
| --- | --- | --- | --- |
| Random forest | 0.6081 | 0.7367 | **+0.1285** |
| Chemprop D-MPNN | 0.6066 | 0.7312 | **+0.1246** |
| GINE graph neural network | 0.6796 | 0.7357 | +0.0560 |

Two things to read off this.

**The optimism is large.** A random split buys roughly 0.125 log units of apparent accuracy on
identical data with identical models — about a 17% reduction in RMSE. Any QSAR result reported
under a random split on a congeneric single-target corpus should be discounted by roughly that
much before being compared with anything else. This is the number this repository exists to
measure, and it is measured rather than asserted.

**The optimism is not architecture-selective.** The forest and the field-standard D-MPNN gain
almost exactly the same amount, and they are tied under *both* splits (0.6081 vs 0.6066 random;
0.7367 vs 0.7312 scaffold). Changing the split changes how good everything looks; it does not
change the ranking.

The GINE network is the exception, and the explanation is deflationary rather than interesting: it
gains least from a random split (+0.056) because it is the weakest model *under* the random split
(0.6796 against ~0.607 for the other two). It has less to gain from near-duplicate analogues
because it is less able to exploit them, which is a statement about its capacity, not about graph
models being more robust to leakage. Reading that gap as an architectural property is exactly the
mistake the correction at the top of this README describes.

## Why the models tie

Three independent lines of evidence point at the data rather than the architectures.

**They fail on the same molecules.** Residual correlations: random forest vs GINE r = 0.852,
random forest vs Chemprop r = 0.787, GINE vs Chemprop r = 0.819. Three quite different inductive
biases produce highly correlated errors, which is the signature of a data-limited problem.

**The graph models are data-limited, not architecturally unsuited.** Test RMSE against training-set
size, subsampling whole scaffold groups so no scaffold is ever split between the retained and
discarded sets:

| training compounds | Random forest | GINE | Chemprop | RF→Chemprop gap |
| --- | --- | --- | --- | --- |
| ~500 | 0.9735 | 1.0885 | 1.0603 | 0.087 |
| ~1,200 | 0.8908 | 1.0256 | 0.9749 | 0.084 |
| ~2,390 | 0.8262 | 0.9060 | 0.8751 | 0.049 |
| ~3,580 | 0.7822 | 0.8549 | 0.8147 | 0.033 |
| ~4,771 | 0.7416 | 0.7666 | 0.7682 | 0.027 |

Both graph models start well behind and close steadily: the forest-to-Chemprop gap shrinks from
0.087 to 0.027 log units as the training set grows roughly tenfold, and the graph curves are still
descending more steeply than the forest's at the largest size available. That is the single best
argument that this negative result would not survive a larger dataset. These
numbers are not directly comparable to the headline table: the curve trains on the training fold
only (4,771 compounds, not train + val's 5,794), under a reduced protocol of two seeds and a
capped epoch budget, and without five-seed prediction averaging.

**The labels have an irreducible noise floor.** The 6,816 compounds come from 1,046 assays across
720 publications. Published inter-laboratory reproducibility for ChEMBL IC50 data is roughly
0.5–0.7 log units, and all three models sit at ~0.73. There is little headroom left for any
architecture to claim.

A fourth, smaller signal: hyperparameter search selected the *smaller* GINE network (hidden 64,
depth 4, 61,889 parameters against 5,794 training molecules). Capacity was not the binding
constraint.

## Where graph models do look better

Test error split by each compound's maximum ECFP4 Tanimoto similarity to the training set:

| max Tanimoto to train | n | Random forest | GINE | Chemprop |
| --- | --- | --- | --- | --- |
| 0.00–0.30 | 8 | 0.8103 | 0.8882 | 0.6468 |
| 0.30–0.40 | 41 | 0.8215 | **0.7813** | **0.7556** |
| 0.40–0.50 | 92 | 0.9795 | **0.8559** | 0.9264 |
| 0.50–0.70 | 452 | 0.7780 | 0.7640 | 0.7734 |
| 0.70–1.00 | 429 | **0.6104** | 0.6666 | 0.6301 |

The forest is strongest where test compounds closely resemble training compounds — it is, in
effect, a sophisticated similarity lookup — and the graph models hold up comparatively better as
compounds get structurally further away. **No individual bin is statistically significant**, every
per-bin interval overlaps, and the lowest bin holds only eight compounds. This is a suggestive
trend, not a demonstrated effect, and it is the result that would most justify a follow-up on a
larger or more diverse dataset.

## The baseline is not a straw man

Feature-block ablations for the forest (single seed):

| Baseline variant | features | RMSE |
| --- | --- | --- |
| ECFP4 only | 2,048 | 0.7477 |
| 12 descriptors only | 12 | 0.9002 |
| **ECFP4 + descriptors** | **2,060** | **0.7367** |
| FCFP4 + descriptors | 2,060 | 0.7601 |
| ECFP6 + descriptors | 2,060 | 0.7412 |

Neither a larger radius nor feature-based invariants beat ECFP4, so the fingerprint choice is not
leaving performance on the table. Descriptors alone are far worse than the fingerprint, so the
signal is genuinely in the substructure rather than in bulk properties — which makes this a fair
graph-versus-substructure comparison rather than a graph-versus-molecular-weight one. The forest
puts 78.9% of its importance on fingerprint bits and 21.1% on the twelve descriptors.

Equally, the graph side is not a straw man: Chemprop's D-MPNN at author defaults is the reference
implementation the field benchmarks against, and it reproduces the hand-rolled GINE network's
result to within 0.005 RMSE. The conclusion does not depend on my implementation.

## The split, and why it can be checked

The split is the part of a QSAR comparison most likely to be quietly wrong, so it is built once and
then made verifiable:

- **Atom-typed Bemis–Murcko scaffolds**, not generic frameworks. For HDAC1 the zinc-binding group
  *is* the pharmacophore, and generic frameworks would merge chemotypes that behave completely
  differently.
- **Acyclic molecules get singleton groups.** `MurckoScaffoldSmiles` returns `""` for ring-free
  molecules; bundling them into one pseudo-scaffold is a classic leak. Only 7 of 6,816 compounds
  are affected, but the handling is explicit.
- **No RNG anywhere in the assignment.** Groups are sorted by descending size with a lexicographic
  tiebreak and filled greedily, so the split is byte-identical across runs, platforms and hash
  seeds — verified identical across three independent runs.
- **One split artifact, hashed.** `data/split_assignment.csv` is SHA-256'd into
  `results/split_manifest.json` (`d25042a5…`). Every model loads it through `src/split_contract.py`,
  which re-verifies the digest, and no model is permitted to import the splitter. Each writes the
  observed hash into its own metrics file, and the comparison refuses to run if they differ. The
  same mechanism checks that all models were scored by the same `src/evaluate.py`.
- **Test-fold access is counted.** The contract counts reads; every model asserts exactly one.

Folds: train 4,771 / val 1,023 / test 1,022 across 3,128 scaffold groups.

**What a scaffold split actually buys on this dataset.** Measured against a seeded random split at
the same fold fractions:

| Split | median test→train Tanimoto | fraction ≥ 0.85 | identical fingerprints |
| --- | --- | --- | --- |
| Random | 0.786 | 25.8% | 78 |
| **Murcko scaffold** | **0.663** | **5.7%** | **2** |

The scaffold split cuts high-similarity pairs 4.5-fold and identical-fingerprint pairs 39-fold. But
its absolute median similarity is still 0.663, because HDAC1's ChEMBL corpus is congeneric — the
dataset's own nearest-neighbour median is ~0.80, since nearly every compound is a zinc-binding
group, a linker and a cap. **A scaffold split on a single-target SAR corpus does not deliver the
chemotype separation the literature's usual 0.3–0.45 figure implies**; that figure describes
diverse multi-target benchmarks. The leakage gate in this repo is therefore calibrated against the
measured random-split control rather than an absolute constant. This was found by a gate failing,
and is written up in `AI_USAGE.md`.

Two test compounds have a *fingerprint* identical to a training compound while being genuinely
different molecules — macrocycles differing by one ring carbon, and a sulfonamide with N and S
transposed — which a folded radius-2 ECFP4 cannot resolve. Those are reported as flags rather than
failures, since the structures are distinct. Note the direction: a fingerprint-identical neighbour
is effectively memorised by the forest but not by a graph model, so this small effect favours the
baseline.

## Method

1. Pull HDAC1 (`CHEMBL325`) IC50 activities from ChEMBL: binding assays, exact relation, nM units,
   non-null pChEMBL, human, and confidence score 9 (direct single-protein assignment).
2. Standardise structures with `rdMolStandardize` (`Cleanup` → `FragmentParent` → `Uncharger` →
   strip stereochemistry), then aggregate to one median pIC50 per parent InChIKey.
3. Drop compounds whose replicate measurements span more than 1.0 log unit as irreconcilable
   (97 of 695 replicated groups). Verify the recomputed pIC50 against ChEMBL's own `pchembl_value`
   to within 0.01 — observed maximum disagreement 0.0055.
4. Featurise with ECFP4 (radius 2, 2048 bits, via `rdFingerprintGenerator.GetMorganGenerator`)
   plus 12 physicochemical descriptors.
5. Split by Murcko scaffold, 70/15/15, deterministically; hash the artifact.
6. Tune a random forest on the validation fold, refit on train + val over five seeds, score once
   on test.
7. Do the same for two graph models — a GINE network and Chemprop's D-MPNN — on the same split,
   same harness, same seed protocol, and compare with paired bootstraps.
8. Re-run everything under a random split to measure how much the evaluation protocol was worth.

**On the architectures.** The GINE network is not a verbatim reimplementation of Gilmer et al.
(2017); it belongs to the same message-passing family but projects edge features to the node
dimension and adds them rather than learning a dense `edge_dim → hidden × hidden` transform, which
at hidden 128 would be a 16,384-output MLP per layer and would overfit badly at this data size.
Chemprop is run at author defaults with no hyperparameter search, because the defaults are what the
field cites and searching them would make it a tuned competitor rather than a reference point.

## Reproducing it

```bash
git clone https://github.com/aadityageddam-ux/hdac1-qsar-gnn.git
cd hdac1-qsar-gnn
uv sync
```

```bash
uv run python scripts/01_fetch_chembl.py
uv run python scripts/02_clean_dataset.py
uv run python scripts/03_build_split.py
uv run python scripts/99_verify.py
```

```bash
uv run python scripts/04_train_baseline.py
uv run python scripts/05_train_gnn.py
uv run python scripts/05b_train_chemprop.py
uv run python scripts/06_compare.py
uv run python scripts/07_diagnostics.py
uv run python scripts/99_verify.py
```

`99_verify.py` is the executable proof. It re-derives every claim above from the committed
artifacts — the split digest, fold disjointness, that every model used the same split and the same
scoring code, that none read the test fold more than once — and it re-runs the evaluation harness
against synthetic inputs whose correct answers are known in advance. It exits non-zero if anything
fails.

**Runtime.** The data phases take a few minutes and the random forest about 14 minutes. The graph
models are the expensive part on CPU: the GINE tuning grid plus five seeds runs about three hours,
Chemprop about 85 minutes, and the diagnostics a further few hours.
`torch.use_deterministic_algorithms(True)` is a meaningful part of that cost and is kept
deliberately — same-seed GINE runs produce bitwise-identical predictions, which is verified as a
gate. No GPU is used or needed.

## Repository layout

```
src/
  chembl_fetch.py     # ChEMBL retrieval + provenance
  standardize.py      # rdMolStandardize pipeline
  featurize.py        # ECFP + descriptor block
  scaffold_split.py   # Murcko scaffolds, deterministic assignment
  split_contract.py   # THE split loader; SHA-256 verified, counts test reads
  evaluate.py         # the only module that computes a metric
  baseline_rf.py      # random forest tuning, multi-seed fitting, ablations
  gnn.py              # molecular graphs, GINE network, training loop
  chemprop_model.py   # Chemprop D-MPNN wrapper at author defaults
  gates.py            # every assertion gate, runnable standalone
  plotting.py         # figures
scripts/              # 01 fetch -> 07 diagnostics, 99 verify
data/                 # committed derived artifacts + provenance.json
results/              # metrics JSON, gate report, figures
report.ipynb          # loads artifacts and renders; computes nothing
```

## Limitations

- **Label noise sets the floor, and all three models are near it.** 1,046 assays across 720
  publications means heterogeneous conditions. Published ChEMBL IC50 reproducibility is ~0.5–0.7
  log units; every model here sits at ~0.73. The reported RMSE is close to a property of the
  dataset rather than of any model.
- **The study is only marginally powered to detect a real difference.** The paired Δ interval is
  roughly ±0.03 log units, and genuine differences between good QSAR methods are often 0.02–0.05.
  "No significant difference" therefore reflects limited power as well as genuine similarity, and
  should not be read as established equivalence.
- **15.5% of measurements were excluded as censored** (`>`, `>=`, `<`, `<=`). These are
  predominantly weak compounds, so excluding them truncates the low-potency tail and makes every
  classification metric look slightly better than it should. The excluded records are kept in
  `data/hdac1_censored.csv` rather than discarded silently.
- **Stereochemistry is stripped** (5,568 stereocentres). ChEMBL annotates it inconsistently across
  papers, so keeping it would create phantom distinct compounds that are really duplicates — a
  leakage channel. This costs real signal: some HDAC inhibitors have an active enantiomer.
- **The scaffold split separates chemotypes less than its name suggests** on a congeneric
  single-target corpus. Quantified above rather than assumed.
- **The similarity-stratified result is suggestive, not significant.** Every per-bin interval
  overlaps.
- **The learning curve uses a reduced protocol** — two seeds and a capped epoch budget, against
  five seeds and a 300-epoch cap for the headline. It answers which way performance trends with
  data volume, not what the exact RMSE is at each size.
- **One target, one assay type, in silico only.** The split-dependence result in particular is
  measured on a single congeneric corpus; whether the asymmetry generalises to structurally
  diverse targets is untested here and is the obvious next experiment.
- **n ≈ 6.8k is small by deep-learning standards.** The most likely reading of this result is that
  the graph models are data-limited rather than architecturally unsuited, which is what the
  learning curve is for.

## AI assistance

Built with substantial AI assistance. `AI_USAGE.md` records the prompts, what was model-generated
versus mine, and the cases where the model was wrong — including a guard that never fired because
`np.std` of identical floats is not exactly zero, and a leakage gate whose threshold encoded an
assumption that measurement refuted.

## Citations

1. Hou P, Li Y, Zhang X, et al. Pluripotent stem cells induced from mouse somatic cells by
   small-molecule compounds. *Science* 2013;341(6146):651–654.
   [PMID 23979017](https://pubmed.ncbi.nlm.nih.gov/23979017/)
2. Cacchiarelli D, Trapnell C, et al. Integrative analyses of human reprogramming reveal dynamic
   nature of induced pluripotency.
   [PMC3943685](https://pmc.ncbi.nlm.nih.gov/articles/PMC3943685/)
3. Yang K, Swanson K, Jin W, et al. Analyzing learned molecular representations for property
   prediction. *J Chem Inf Model* 2019;59(8):3370–3388.
   [PMID 31361484](https://pubmed.ncbi.nlm.nih.gov/31361484/)
4. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for quantum
   chemistry. *ICML* 2017. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
5. Duvenaud D, Maclaurin D, Aguilera-Iparraguirre J, et al. Convolutional networks on graphs for
   learning molecular fingerprints. *NeurIPS* 2015.
   [arXiv:1509.09292](https://arxiv.org/abs/1509.09292)
6. Hu W, Liu B, Gomes J, et al. Strategies for pre-training graph neural networks. *ICLR* 2020.
   [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
7. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks.
   *J Med Chem* 1996;39(15):2887–2893. [PMID 8709122](https://pubmed.ncbi.nlm.nih.gov/8709122/)
8. Kramer C, Kalliokoski T, Gedeck P, Vulpetti A. The experimental uncertainty of heterogeneous
   public Ki data. *J Med Chem* 2012;55(11):5165–5173.
   [PMID 22643060](https://pubmed.ncbi.nlm.nih.gov/22643060/)
9. Kalliokoski T, Kramer C, Vulpetti A, Gedeck P. Comparability of mixed IC50 data — a statistical
   analysis. *PLoS One* 2013;8(4):e61007.
   [PMID 23613770](https://pubmed.ncbi.nlm.nih.gov/23613770/)
10. Zdrazil B, Felix E, Hunter F, et al. The ChEMBL Database in 2023. *Nucleic Acids Res*
    2024;52(D1):D1180–D1192. [PMID 37933841](https://pubmed.ncbi.nlm.nih.gov/37933841/)
11. RDKit: Open-source cheminformatics. <https://www.rdkit.org>

**Data source.** ChEMBL target `CHEMBL325` (histone deacetylase 1, *Homo sapiens*, UniProt
Q13547), retrieved 2026-09-26 via `chembl_webresource_client`. Exact query predicates and
per-stage record counts are in `data/provenance.json`. The query pattern follows the
[TeachOpenCADD T001 talktorial](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html).
