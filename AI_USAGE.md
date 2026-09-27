# AI Usage

**Model used:** Claude Opus 5, via Claude Code (Anthropic's CLI agent), running in agent mode
with subagents for the data phase and for an independent verification pass.

This project was built with heavy AI assistance. The sections below record what I asked for,
what the model did, where it was wrong, and how each error was caught — including two cases
where AI-written code encoded an assumption that measurement later refuted.

## Prompts

1. > HDAC1 QSAR + GNN Pipeline — Build Brief
   >
   > OVERVIEW
   > Build a reproducible pipeline predicting small molecule bioactivity against
   > HDAC1, a chromatin-modifying enzyme used across published chemical
   > reprogramming protocols. [...] Pull public bioactivity data from ChEMBL,
   > featurize compounds with RDKit, and compare a classical machine learning
   > baseline against a graph neural network, validated with a scaffold split
   > rather than a random split.

   The full brief ran to roughly 120 lines and specified the scientific motivation, the data
   and method steps, seven build phases, an agent-orchestration strategy (plan mode first,
   phases 2–3 as one subagent, phases 4–5 strictly sequential, a separate verification
   subagent per phase), the relevance framing for the target role, and explicit risk and
   fallback clauses covering the GNN stalling, class imbalance, and the possibility that the
   GNN simply loses.

   In response the model entered plan mode, queried the live ChEMBL API to measure the dataset
   before designing around it, ran a background design agent, asked four clarifying questions,
   and wrote a plan file that was approved before any code was written.

2. Four clarifying questions were answered during planning, selecting: a `src/` package plus
   numbered scripts over a single narrative notebook; a three-way scaffold split with
   validation-fold tuning and five-seed multi-seed reporting over fixed defaults; inclusion of
   the learning-curve and similarity-stratification diagnostics; and building locally with the
   commit left unpushed for review.

3. > continue

4. One further decision was escalated mid-build, when the leakage gates failed — see
   Interaction 2 below.

## What the AI got wrong, and how it was caught

### Interaction 1: a constant-input guard that never fired

`src/evaluate.py` guarded the Spearman and Pearson correlations against a constant prediction
vector, so that a collapsed model would score `NaN` rather than a meaningless correlation:

```python
if np.std(y_pred) == 0 or np.std(y_true) == 0:
    return float("nan")
```

A property test of the harness surfaced a warning that should have been impossible if the
guard worked:

```
ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  return float(stats.spearmanr(y_true, y_pred).statistic)
```

The cause is that `np.std` of an array of *identical* floats is not exactly zero:

```
np.std(pred) == np.float64(1.7763568394002505e-15) | == 0 -> False
```

The mean of the array is generally not exactly representable in binary floating point, so the
two-pass variance leaves roughly 1e-15 of residue, and the `== 0` comparison is never true. The
test had originally been written only to assert that the returned value was `NaN` — which it
was, because SciPy independently detected the constant input and returned `NaN` itself. The
guard was dead code and the test passed anyway.

The fix replaces the standard deviation with a peak-to-peak comparison, which is a max minus a
min and involves no arithmetic, so it is exactly zero precisely when every element is identical:

```python
def _is_constant(a: np.ndarray) -> bool:
    return a.size == 0 or float(np.ptp(a)) == 0.0
```

This mattered beyond the warning. A near-constant prediction vector — a GNN that collapsed to
predicting the mean, or a bootstrap resample of a near-constant label — would have slipped past
the broken guard and been scored with a garbage correlation instead of an honest `NaN`.

### Interaction 2: a gate threshold that encoded an unvalidated assumption

The approved plan specified a leakage gate requiring that the median maximum ECFP4 Tanimoto
similarity from each test compound to the training set fall in 0.30–0.45, failing above 0.60.
On the real split, four gates failed:

```
FAIL  L8_no_identical_to_train    observed=2 at sim=1.0      expected=0 at sim=1.0
FAIL  L10_median_similarity       observed=test median=0.6625 expected=<= 0.6 (expected (0.3, 0.45))
FAIL  L11_no_identical_to_train_val observed=6 at sim=1.0    expected=0 at sim=1.0
FAIL  L11c_median_similarity_val  observed=val median=0.7021  expected=<= 0.6
```

Both failures turned out to be the gate being wrong rather than the split being broken, but
that only became clear after investigating rather than adjusting.

**The `sim == 1.0` pairs are not duplicates.** Inspecting all eight showed genuinely distinct
molecules that a folded 2048-bit binary ECFP4 at radius 2 cannot separate — one pair of
macrocyclic peptides differing by a single ring carbon, one sulfonamide with N and S transposed
between the two aryl rings, and six linker homologues differing by exactly one methylene.
Adding a methylene mid-chain creates no new radius-2 atom environment, so the bit set is
unchanged. All eight pairs have distinct InChIKeys, distinct canonical SMILES and distinct
Murcko scaffolds, so standardisation and InChIKey aggregation had worked correctly. The gate
was using Tanimoto identity as a proxy for structural identity, and on this dataset the proxy
had an 8/8 false-positive rate. It was reformulated to test canonical-SMILES identity directly,
with the fingerprint-collision count demoted to a reported flag.

**The 0.30–0.45 band was the wrong reference class.** Measuring a seeded random split at the
same fold fractions gave the missing denominator:

| split | median test→train Tanimoto | fraction ≥ 0.85 | identical fingerprints |
| --- | --- | --- | --- |
| random | 0.786 | 25.8% | 78 |
| Murcko scaffold | 0.663 | 5.7% | 2 |

The dataset's own nearest-neighbour median is about 0.80: HDAC1's ChEMBL corpus is congeneric,
assembled from 781 SAR papers around one pharmacophore. The scaffold split was in fact working —
it cut high-similarity pairs 4.5-fold and identical-fingerprint pairs 39-fold — but no
scaffold-based scheme can reach 0.30–0.45 on data this homogeneous. That band describes a
diverse multi-target benchmark, and I had asserted it from prior expectation rather than
measurement.

The gate is now expressed relative to the measured random-split control (it must sit a clear
margin below it), which is the quantity it actually cares about and is self-calibrating across
targets. **The split itself was not changed** — `split_assignment.csv` hashes to the same
`d25042a5…` before and after — so this was a recalibration of the check, not of the result.

I deliberately did not let the subagent resolve this on its own: it was instructed to stop, and
the choice between recalibrating the gate and redesigning the split was escalated to me, because
choosing a split because it produces a desired number is the exact failure the gate existed to
prevent.

### Interaction 3: unwired code that could not have run

A review pass found that `src/gates.py` defined `run_feature_gates()`, which calls
`build_features(...)`, while the module's import line read only:

```python
from src.featurize import max_similarity_to_reference
```

The function was never called from anywhere, so nothing raised and every run appeared clean —
but the two feature gates specified in the plan (matrix shape, and a finiteness check guarding
against `MolLogP` returning `inf`) were not actually being executed on the data. The import was
completed and the gates wired into the verification pass, where they now run and pass.

### Interaction 4: two checks that tested the wrong thing

Two more cases of the same shape as Interaction 2 — the check being wrong rather than the code —
both caught by running the verification pass rather than by reading it.

The first: `scripts/99_verify.py` asserted that neither model script imports the splitter, since
importing it would let a model recompute the folds instead of loading the verified artifact. It
tested this with a substring search:

```python
check(f"{script} does not import scaffold_split", "scaffold_split" not in source)
```

Both scripts failed. Neither imports it — but both *docstrings* contain the sentence "this script
never imports `scaffold_split`", so the check was flagging its own documentation. It now walks the
AST for `Import` and `ImportFrom` nodes, which is the stricter test as well as the correct one,
because a comment cannot satisfy it.

The second: the learning-curve figure drew both confidence bands in the same blue, so the GNN's
band did not match its own orange line. The colour lookup took the first word of each model label:

```python
colour = MODEL_COLOURS.get(label.lower().split()[0], None)
```

The first words are `random` and `gine`, neither of which is a key, so both bands resolved to
`None` and Matplotlib defaulted them to the same colour. The fix resolves colours by substring
and then reads the realised colour back off the drawn line, so a band can never disagree with the
line it belongs to. The same figure's title also still claimed three seeds after the protocol was
reduced to two; it is now generated from the seed count actually used.

Neither bug would have changed a number. Both would have misled a reader — one by asserting a
safety property that was never really being tested, the other by mislabelling a chart.

## What was mine vs. AI-generated

Effectively all of the code in `src/` and `scripts/` was written by Claude, as were the first
drafts of this file and the README. My contribution was the brief, the four planning decisions,
the escalated split-versus-gate decision in Interaction 2, and the review that rejected the
AI's initial instinct to make failing gates pass.

The substantive judgement calls are recorded where they were made rather than smoothed over: the
classification threshold was pre-registered at the conventional 100 nM cut before any model was
scored and not moved afterwards; the failing gates were investigated rather than relaxed; and
the comparison verdict in the README is generated programmatically from the confidence interval
in `results/comparison.json` rather than written by hand, so the prose cannot drift from the
numbers it describes.
