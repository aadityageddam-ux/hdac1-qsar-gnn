# HDAC1 bioactivity: a fingerprint random forest vs. a message-passing graph neural network

**Does a message-passing graph neural network learn anything about HDAC1 inhibition that a
2048-bit ECFP4 random forest does not, when both are tested on chemical scaffolds neither has
seen?**

**Result: no — the two are statistically indistinguishable.** On a leak-free Murcko scaffold
split of 6,816 compounds, the random forest reaches a test RMSE of **0.7367** pIC50 log units
and the GNN **0.7357**. The paired bootstrap on the difference gives **+0.0010, 95% CI
[−0.0307, +0.0293], p = 0.936** — and no significant difference on any of the nine metrics
measured.

This is a negative result and is reported as such. It is not, however, a null one: both models
beat a 1-nearest-neighbour Tanimoto lookup by a wide margin (RMSE 0.944), so the dataset is not
merely memorisable by similarity, and the two models' residuals correlate at r = 0.852, which
says they are defeated by the same molecules. At this data size the ceiling is the data, not the
architecture.

**The most transferable finding is about the split, not the models.** Re-run under a random
split, the same two models on the same data give RF 0.6081 and GNN 0.6796 — the forest appears
to *beat* the GNN by 0.07, a clean and publishable-looking win. Under the leak-free scaffold
split they tie. The choice of split, not the choice of architecture, decides what this
experiment appears to show.

## The question

Chemical reprogramming replaces or supplements the Yamanaka factors with small molecules to
induce pluripotency without genetic modification. HDAC inhibitors recur across published
cocktails, and HDAC1 acts directly on chromatin — making it a mechanistically precise example of
epigenetic reprogramming rather than one of the signalling-pathway compounds that enable
reprogramming indirectly. It is also an established oncology target, so ChEMBL carries a large,
well-curated bioactivity dataset for it.

That makes it a good setting for a narrower methodological question: on a real single-target
QSAR dataset of a few thousand compounds, does operating on the molecular graph buy anything
over a well-tuned fingerprint baseline, once you remove the scaffold overlap that random splits
leave behind?

## The result

Test fold, 1,022 compounds. Every model is scored by the same code (`src/evaluate.py`), on the
same split, with the same 2,000 bootstrap resamples. Both models were tuned on the validation
fold only, refit on train + val, and run over five seeds.

| Model | RMSE ↓ | MAE ↓ | R² ↑ | Spearman ↑ | ROC-AUC ↑ | PR-AUC ↑ | MCC ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Mean predictor | 1.1060 | 0.9007 | −0.017 | — | 0.500 | 0.309 | 0.000 |
| Majority class | 1.2015 | 0.9499 | −0.200 | — | 0.500 | 0.309 | 0.000 |
| Random scores | 2.2340 | 1.8290 | −3.147 | 0.009 | 0.483 | 0.293 | 0.011 |
| 1-NN Tanimoto | 0.9440 | 0.6530 | 0.259 | 0.657 | 0.818 | 0.659 | 0.466 |
| **Random forest** (ECFP4 + 12 descriptors) | **0.7367** | 0.5640 | 0.549 | **0.7594** | **0.8779** | **0.7784** | **0.5636** |
| **GINE graph neural network** | **0.7357** | **0.5535** | **0.550** | 0.7456 | 0.8670 | 0.7563 | 0.5250 |
| **Δ (RF − GNN), paired bootstrap** | **+0.0010** <br> [−0.0307, +0.0293] | +0.0105 | −0.0012 | +0.0138 | +0.0109 | +0.0221 | +0.0386 |

With 95% bootstrap intervals on the two headline numbers: RF RMSE 0.7367 [0.6974, 0.7742],
GNN 0.7357 [0.6938, 0.7804]. Classification metrics use a threshold of pIC50 ≥ 7.0 (IC50 ≤
100 nM), pre-registered as the conventional medicinal-chemistry potency cut before any model was
scored; the test-fold positive rate is 0.309.

**The Δ row is the headline, not the two intervals.** Two overlapping marginal confidence
intervals do not establish that two models are indistinguishable. The paired bootstrap scores
both models on the same resamples and takes the difference, which is the correct test — and here
it confirms the tie rather than merely failing to reject it.

### One result needs two numbers

The headline RMSE for each model is the mean prediction across five seeds. That treats both
models identically, but it hides something worth stating:

| | single model (mean ± sd over 5 seeds) | 5-seed averaged prediction |
| --- | --- | --- |
| Random forest | 0.7370 ± 0.0010 | 0.7367 |
| GINE GNN | **0.7744 ± 0.0120** | **0.7357** |

A single GNN is clearly worse than a single random forest. Averaging five independently
initialised GNNs recovers essentially all of that gap, because the GNN has real seed-to-seed
variance while a 1,000-tree random forest is already an ensemble and gains nothing from being
averaged again. So "the GNN matches the baseline" is true only of the ensembled GNN. A
practitioner choosing one model to deploy, on this dataset, should take the forest: it is
cheaper, it trains in minutes rather than hours, and it is effectively deterministic.

### Interpretation

**Why the GNN does not win here.** Three pieces of evidence point the same way:

- **Residual correlation between the two models is r = 0.852.** They fail on the same compounds.
  That is the signature of a data-limited problem rather than an architecture-limited one.
- **The tuning search selected the *smaller* network** — hidden 64, depth 4, dropout 0.2, 61,889
  parameters against 5,794 training molecules. Capacity was not the binding constraint; data was.
- **The labels have an irreducible noise floor.** The 6,816 compounds come from 1,046 assays
  across 720 publications. Published inter-laboratory reproducibility for ChEMBL IC50 data is
  roughly 0.5–0.7 log units, and both models sit at ~0.74. There is not much headroom left for
  any architecture to claim.

**Where the GNN does look better.** Test error split by each compound's maximum ECFP4 Tanimoto
similarity to the training set:

| max Tanimoto to train | n | Random forest | GINE GNN |
| --- | --- | --- | --- |
| 0.00–0.30 | 8 | 0.8103 | 0.8882 |
| 0.30–0.40 | 41 | 0.8215 | **0.7813** |
| 0.40–0.50 | 92 | 0.9795 | **0.8559** |
| 0.50–0.70 | 452 | 0.7780 | **0.7640** |
| 0.70–1.00 | 429 | **0.6104** | 0.6666 |

The pattern is the one you would predict: the forest is strongest where test compounds closely
resemble training compounds — it is, in effect, a sophisticated similarity lookup — and the GNN
holds up better as compounds get structurally further away. **No individual bin is statistically
significant**, the per-bin confidence intervals all overlap, and the lowest bin holds only eight
compounds. This is a suggestive trend consistent across three adjacent bins, not a demonstrated
effect, and it is reported that way. It is the result that would most justify a follow-up on a
larger or more diverse dataset.

The GNN is also better on the 89 test compounds with two or more independent measurements
(0.7701 vs 0.8116) — the subset with the least label noise — which is weakly consistent with the
same story.

**The GNN is data-limited, not architecturally unsuited.** Test RMSE against training-set size,
subsampling whole scaffold groups so that no scaffold is ever split between the retained and
discarded sets:

| training compounds | Random forest | GINE GNN | gap |
| --- | --- | --- | --- |
| ~500 | 0.9735 | 1.0885 | 0.115 |
| ~1,200 | 0.8908 | 1.0256 | 0.135 |
| ~2,390 | 0.8262 | 0.9060 | 0.080 |
| ~3,580 | 0.7822 | 0.8549 | 0.073 |
| ~4,771 | 0.7416 | 0.7666 | **0.025** |

The GNN is far behind at small sample sizes and closes steadily: the gap shrinks from 0.135 to
0.025 log units as the training set grows roughly fourfold, and its curve is still descending
more steeply than the forest's at the largest size available. That is the signature of a model
limited by data volume rather than by architecture, and it is the single best argument that this
negative result would not survive a larger dataset.

These numbers are not directly comparable to the headline table: the curve trains on the training
fold only (4,771 compounds, not train + val's 5,794), under a reduced protocol of two seeds and a
capped epoch budget, and without the five-seed prediction averaging. It answers which way
performance trends with data, not what the exact RMSE is at each size.

**A random split would have produced the opposite conclusion.** The same two models, the same
data, the same fold fractions, differing only in how compounds were assigned:

| Model | random split | scaffold split | optimism |
| --- | --- | --- | --- |
| Random forest | 0.6081 | 0.7367 | **+0.1285** |
| GINE GNN | 0.6796 | 0.7357 | **+0.0560** |

A random split flatters the random forest more than twice as much as it flatters the GNN
(+0.129 vs +0.056 log units). That asymmetry is exactly what you would expect if the forest is
substantially a similarity lookup: a random split scatters near-duplicate analogues across train
and test, and a fingerprint model exploits them more effectively than a graph model does. Under
that evaluation the forest would have looked 0.07 log units *better* than the GNN, and the
honest conclusion — that the two are indistinguishable — would have been invisible. This is the
clearest argument in the repository for why the scaffold split was worth the trouble.

**The baseline is not a straw man.** Feature-block ablations (single seed):

| Baseline variant | features | RMSE |
| --- | --- | --- |
| ECFP4 only | 2,048 | 0.7477 |
| 12 descriptors only | 12 | 0.9002 |
| **ECFP4 + descriptors** | **2,060** | **0.7367** |
| FCFP4 + descriptors | 2,060 | 0.7601 |
| ECFP6 + descriptors | 2,060 | 0.7412 |

Neither a larger radius nor feature-based invariants beat ECFP4, so the fingerprint choice is not
leaving performance on the table. Descriptors alone are far worse than the fingerprint, so the
signal is genuinely in the substructure rather than in bulk properties — which is what makes this
a fair graph-versus-substructure comparison rather than a graph-versus-molecular-weight one.
The forest puts 78.9% of its importance on fingerprint bits and 21.1% on the twelve descriptors.

## The split, and why it can be checked

The split is the part of a QSAR comparison most likely to be quietly wrong, so it is built once
and then made verifiable:

- **Atom-typed Bemis–Murcko scaffolds**, not generic frameworks. For HDAC1 the zinc-binding group
  *is* the pharmacophore, and generic frameworks would merge chemotypes that behave completely
  differently.
- **Acyclic molecules get singleton groups.** `MurckoScaffoldSmiles` returns `""` for ring-free
  molecules; bundling them into one pseudo-scaffold is a classic leak. Only 7 of 6,816 compounds
  are affected here, but the handling is explicit.
- **No RNG anywhere in the assignment.** Groups are sorted by descending size with a lexicographic
  tiebreak and filled greedily, so the split is byte-identical across runs, platforms and hash
  seeds. It verified as identical across three independent runs.
- **One split artifact, hashed.** `data/split_assignment.csv` is SHA-256'd into
  `results/split_manifest.json` (`d25042a5…`). Both models load it through `src/split_contract.py`,
  which re-verifies the digest, and neither model is permitted to import the splitter. Each writes
  the observed hash into its own metrics file, and the comparison refuses to run if they differ.
  The same mechanism checks that both models were scored by the same `src/evaluate.py`.
- **Test-fold access is counted.** The contract counts reads; both models assert exactly one.

Folds: train 4,771 / val 1,023 / test 1,022 across 3,128 scaffold groups.

**An honest caveat about what a scaffold split buys on this dataset.** Measured against a seeded
random split at the same fold fractions:

| Split | median test→train Tanimoto | fraction ≥ 0.85 | identical fingerprints |
| --- | --- | --- | --- |
| Random | 0.786 | 25.8% | 78 |
| **Murcko scaffold** | **0.663** | **5.7%** | **2** |

The scaffold split cuts high-similarity pairs 4.5-fold and identical-fingerprint pairs 39-fold.
But its absolute median similarity is still 0.663, because HDAC1's ChEMBL corpus is congeneric —
the dataset's own nearest-neighbour median is ~0.80, since nearly every compound is a
zinc-binding group, a linker and a cap. **A scaffold split on a single-target SAR corpus does not
deliver the chemotype separation the literature's usual 0.3–0.45 figure implies**, and that
figure describes diverse multi-target benchmarks instead. The leakage gate in this repo is
therefore calibrated against the measured random-split control rather than an absolute constant.
This was found by a gate failing, and is written up in `AI_USAGE.md`.

Two test compounds have a *fingerprint* identical to a training compound while being genuinely
different molecules — macrocycles differing by one ring carbon, and a sulfonamide with N and S
transposed — which a folded radius-2 ECFP4 cannot resolve. Those are reported as flags rather
than failures, since the structures are distinct. Note the direction: a fingerprint-identical
neighbour is effectively memorised by the forest but not by the GNN, so this small effect favours
the baseline.

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
7. Do the same for a GINE message-passing network on molecular graphs — same split, same harness,
   same seed protocol — and compare with a paired bootstrap.

**On the architecture.** This is a `GINEConv` network, not a verbatim reimplementation of Gilmer
et al. (2017). GINEConv belongs to the same message-passing family and consumes bond features
directly, but it projects edge features to the node dimension and adds them rather than learning
a dense `edge_dim → hidden × hidden` transform. At hidden 128 that transform is a 16,384-output
MLP per layer, which on a few thousand molecules would overfit badly and would make a negative
result attributable to the architecture rather than to the data. Saying "GINE, and here is why"
seemed more useful than claiming Gilmer 2017 and quietly implementing something else.

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
uv run python scripts/06_compare.py
uv run python scripts/07_diagnostics.py
uv run python scripts/99_verify.py
```

`99_verify.py` is the executable proof. It re-derives every claim above from the committed
artifacts — the split digest, fold disjointness, that both models used the same split and the
same scoring code, that neither read the test fold more than once — and it re-runs the evaluation
harness against synthetic inputs whose correct answers are known in advance. It exits non-zero if
anything fails.

**Runtime.** The data phases take a few minutes. The random forest takes ~14 minutes on CPU. The
GNN is the expensive part: ~5 s/epoch on this dataset, so tuning plus five seeds takes roughly
three hours, and the diagnostics another 75 minutes. `torch.use_deterministic_algorithms(True)` is a meaningful part of that cost and is kept
deliberately — same-seed runs produce bitwise-identical predictions, which is verified as a gate.
No GPU is used or needed.

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
  gates.py            # every assertion gate, runnable standalone
  plotting.py         # figures
scripts/              # 01 fetch -> 07 diagnostics, 99 verify
data/                 # committed derived artifacts + provenance.json
results/              # metrics JSON, gate report, figures
report.ipynb          # loads artifacts and renders; computes nothing
```

## Limitations

- **Label noise sets the floor, and both models are near it.** 1,046 assays across 720
  publications means heterogeneous conditions. Published ChEMBL IC50 reproducibility is ~0.5–0.7
  log units; both models sit at ~0.74. The reported RMSE is close to a property of the dataset
  rather than of either model.
- **15.5% of measurements were excluded as censored** (`>`, `>=`, `<`, `<=`). These are
  predominantly weak compounds, so excluding them truncates the low-potency tail and makes every
  classification metric here look slightly better than it should. The excluded records are kept in
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
- **One target, one assay type, in silico only.** Nothing here was tested experimentally, and
  nothing here supports a claim about any compound's behaviour in cells.
- **n ≈ 6.8k is small by deep-learning standards.** The most likely reading of this result is that
  the GNN is data-limited rather than architecturally unsuited, which is exactly what the learning
  curve is for.

## AI assistance

Built with substantial AI assistance. `AI_USAGE.md` records the prompts, what was model-generated
versus mine, and three cases where the model was wrong — including a guard that never fired
because `np.std` of identical floats is not exactly zero, and a leakage gate whose threshold
encoded an assumption that measurement refuted.

## Citations

1. Hou P, Li Y, Zhang X, et al. Pluripotent stem cells induced from mouse somatic cells by
   small-molecule compounds. *Science* 2013;341(6146):651–654.
   [PMID 23979017](https://pubmed.ncbi.nlm.nih.gov/23979017/)
2. Cacchiarelli D, Trapnell C, et al. Integrative analyses of human reprogramming reveal dynamic
   nature of induced pluripotency.
   [PMC3943685](https://pmc.ncbi.nlm.nih.gov/articles/PMC3943685/)
3. Gilmer J, Schoenholz SS, Riley PF, Vinyals O, Dahl GE. Neural message passing for quantum
   chemistry. *ICML* 2017. [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)
4. Duvenaud D, Maclaurin D, Aguilera-Iparraguirre J, et al. Convolutional networks on graphs for
   learning molecular fingerprints. *NeurIPS* 2015.
   [arXiv:1509.09292](https://arxiv.org/abs/1509.09292)
5. Hu W, Liu B, Gomes J, et al. Strategies for pre-training graph neural networks. *ICLR* 2020.
   [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
6. Bemis GW, Murcko MA. The properties of known drugs. 1. Molecular frameworks.
   *J Med Chem* 1996;39(15):2887–2893. [PMID 8709122](https://pubmed.ncbi.nlm.nih.gov/8709122/)
7. Kramer C, Kalliokoski T, Gedeck P, Vulpetti A. The experimental uncertainty of heterogeneous
   public Ki data. *J Med Chem* 2012;55(11):5165–5173.
   [PMID 22643060](https://pubmed.ncbi.nlm.nih.gov/22643060/)
8. Kalliokoski T, Kramer C, Vulpetti A, Gedeck P. Comparability of mixed IC50 data — a statistical
   analysis. *PLoS One* 2013;8(4):e61007.
   [PMID 23613770](https://pubmed.ncbi.nlm.nih.gov/23613770/)
9. Zdrazil B, Felix E, Hunter F, et al. The ChEMBL Database in 2023. *Nucleic Acids Res*
   2024;52(D1):D1180–D1192. [PMID 37933841](https://pubmed.ncbi.nlm.nih.gov/37933841/)
10. RDKit: Open-source cheminformatics. <https://www.rdkit.org>

**Data source.** ChEMBL target `CHEMBL325` (histone deacetylase 1, *Homo sapiens*, UniProt
Q13547), retrieved 2026-09-26 via `chembl_webresource_client`. Exact query predicates and
per-stage record counts are in `data/provenance.json`. The query pattern follows the
[TeachOpenCADD T001 talktorial](https://projects.volkamerlab.org/teachopencadd/talktorials/T001_query_chembl.html).
