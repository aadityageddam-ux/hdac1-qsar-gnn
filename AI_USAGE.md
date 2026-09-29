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

4. > Be brutally honest and grade this project to show a hiring manager, founder, VC as a work
   > sample, especially for Genesis Molecular AI, and if it has holes if we can improve it or
   > scrap it

   This produced the assessment and the four-item remediation list described below.

5. > do items all items one at a time

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

## The second pass: what a critical review changed

After the first version was complete and verified, I asked for a deliberately harsh assessment of
the repository as a work sample. The review graded it B+ and argued that the craft was strong but
the science was derivative, and it identified four specific gaps. All four were then closed:

1. **No field-standard graph baseline.** The original comparison used only a hand-rolled GINE
   network, which leaves open the objection that the graph side lost because it was weak.
   Chemprop's D-MPNN was added (`scripts/05b_train_chemprop.py`). It reproduced the scaffold-split
   tie to within 0.005 RMSE — and it refuted the headline claim the README had been restructured
   around. See Interaction 5.
2. **The lede was buried.** The model comparison reproduces what the field already believes; the
   quantified split-dependence is the part that generalises. The README was restructured around it.
3. **One target is an anecdote.** `scripts/10_multitarget.py` re-runs the core experiment on HDAC6
   and hERG as well, testing whether the effect replicates within the enzyme family and across a
   completely different target class.
4. **This file was unbalanced** - see below.

Two smaller things the review got right and I acted on: the study is only marginally powered to
detect a real difference, which is now stated in Limitations rather than left implicit; and the
similarity-stratified result is suggestive rather than significant, which the README now says
plainly instead of leaning on it.

One claim in the review I checked and partially rejected. It assumed hERG would be a structurally
diverse contrast to HDAC1's congeneric corpus. Measuring nearest-neighbour similarity across four
candidate targets showed all of them are similarly congeneric (median 0.79-0.81), so hERG was kept
for being a different target class, not for being more diverse - and the observation that this is
systematic across ChEMBL single-target corpora, rather than an HDAC1 quirk, became a finding in
its own right.

## Interaction 5: adding the standard baseline refuted my own headline

This is the most instructive thing that happened in the project, and it happened because of the
critical review rather than in spite of it.

After the first pass, the README's headline finding was that a random split flatters a fingerprint
model substantially more than a graph model — measured as +0.1285 log units of optimism for the
random forest against +0.0560 for the GINE network, a better-than-2× asymmetry, large enough that a
random split reversed which model appeared to win. I found that result genuinely exciting and built
the restructured README around it.

The review's first criticism was that the project never ran Chemprop, the D-MPNN the field actually
benchmarks against, leaving open the objection that the graph side lost because it was a weak graph
model. I added Chemprop expecting it to corroborate the asymmetry. It did the opposite:

| Model | random split | scaffold split | optimism |
| --- | --- | --- | --- |
| Random forest | 0.6081 | 0.7367 | +0.1285 |
| **Chemprop D-MPNN** | **0.6066** | **0.7312** | **+0.1246** |
| GINE (mine) | 0.6796 | 0.7357 | +0.0560 |

Chemprop's optimism is within 0.004 of the forest's. The asymmetry was a property of my weaker
network, not of graph models — and the deflationary explanation is visible in the same table: GINE
gains least from a random split because it is *worst* under the random split, so it has less
ability to exploit the near-duplicate analogues a random split leaves lying around. I had read a
capacity limitation as an architectural virtue.

What survived is smaller but real, and is now the headline: a random split buys ~0.125 log units
(~17% RMSE) of apparent accuracy on identical data and identical models, regardless of
architecture.

The correction is kept visible at the top of the README rather than quietly rewritten, for two
reasons. The overturned claim is a good example of the specific failure mode the project is about —
drawing an architectural conclusion from an evaluation artifact — and I would rather a reader see
that I caught it than discover the earlier version in the git history. It is also the clearest
demonstration in this repository of why the "just add the standard baseline" criticism was worth
acting on: it did not strengthen my result, it deleted it.

## Interaction 6: my own bug reversed a headline finding, twice

Interaction 5 recorded that adding Chemprop overturned the claim that random splits flatter
fingerprint models more than graph models. Extending the experiment to three targets overturned it
again, in the opposite direction, and this time the cause was a bug I had written.

`scripts/07_diagnostics.py` trained the neural models on the random split for an epoch budget that
had been selected on the *scaffold* split:

```python
cp_epochs = int(cp_metrics["config"]["selected_epochs"])   # chosen on the SCAFFOLD split
cp_model, _, _, _ = CP.train_once(
    pd.concat([r_train, r_val], ignore_index=True), None, seed=0,
    max_epochs=cp_epochs, use_early_stopping=False,
)
```

The random split is an easier problem, so it supports a longer useful training run. Capping the
neural models at the scaffold-derived epoch count under-trained them there, depressing their
random-split scores and therefore their measured optimism. The random forest has no epoch
selection, so its number was unbiased — meaning the bug biased *only* the neural rows, in exactly
the direction that made the forest look uniquely sensitive to leakage. It manufactured the
asymmetry I had reported.

`scripts/10_multitarget.py` was written later and independently, and selects the epoch count on
each split's own validation fold. When its numbers disagreed with the diagnostics run, that
disagreement is what exposed the bug: the two scripts were measuring the same quantity and getting
different answers, so one of them had to be wrong.

With the fix, the asymmetry reverses and replicates on all three targets — the forest captures only
0.66–0.76× the optimism the D-MPNN does. The mechanism is sensible in hindsight: a random split
leaves near-duplicate analogues in the training set, and a higher-capacity graph model exploits
them better than a bagged forest over a fixed fingerprint.

Three things worth drawing out, since this is the most useful failure in the project:

- **The bug was in the measurement, not the model.** Every model trained correctly; the comparison
  between them was rigged by a protocol detail that looked like an implementation shortcut.
- **It was caught by redundancy, not by review.** Nobody read the line and spotted it. Two scripts
  computed the same quantity under different assumptions and disagreed, and the disagreement was
  not dismissible.
- **The original claim was the one I most wanted to be true.** It was the finding the README had
  been restructured around, and it was the thing I would have led with in outreach. That is exactly
  the kind of result that deserves a second implementation before it is believed.

## What was mine vs. AI-generated

**Claude wrote effectively all of the code** in `src/` and `scripts/`, and the first drafts of this
file and the README. That should be read literally: I did not type these modules.

What was mine was the specification and the decisions, which in a project like this is where most
of the load sits:

- **The brief.** The scientific framing, the choice of HDAC1 and why it beats the signalling-pathway
  compounds as an example, the seven build phases, the requirement that a verification pass be run
  by an agent that did not build the phase being checked, and explicit risk clauses for the GNN
  stalling and for class imbalance. The methodological spine of the project is in the brief.
- **Four design decisions** taken during planning: scripts over a single notebook, a validation-fold
  tuning protocol with multi-seed reporting over fixed defaults, inclusion of the learning-curve and
  similarity diagnostics, and building locally with the commit left unpushed for review.
- **The escalated call in Interaction 2.** When the leakage gates failed, the model's first instinct
  was to resolve it; I stopped it and made the call myself, because choosing a split on the basis
  that it produces a desired number is exactly the failure the gate existed to prevent.
- **Commissioning the adversarial review** that produced the second pass above, and deciding which
  of its criticisms to act on.

**What this does not entitle me to claim.** I did not derive the D-MPNN, implement the bootstrap,
or independently discover that `np.std` of identical floats is non-zero. If you are evaluating this
as evidence of engineering skill, weight it accordingly - and ask me to explain any part of
`src/evaluate.py` or `src/scaffold_split.py`, which is the fair test of whether I understand what is
in this repository.

The substantive judgement calls are recorded where they were made rather than smoothed over: the
classification threshold was pre-registered at the conventional 100 nM cut before any model was
scored and not moved afterwards; the failing gates were investigated rather than relaxed; the
multiple-comparisons problem is stated rather than exploited (1 of 27 tests reached p < 0.05, which
is what 27 tests produce by chance, and the README says so); and the comparison verdict is generated
programmatically from the confidence interval in `results/comparison.json` rather than written by
hand, so the prose cannot drift from the numbers it describes.
