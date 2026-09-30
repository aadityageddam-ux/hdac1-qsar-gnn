# AI Usage

**Model:** Claude Opus 5, via Claude Code.

I built this with a lot of AI help. I'd rather say exactly how much than be vague about it.

This file covers what I asked for, what the model actually did, and the four things that went
wrong. Two of those were bugs that reversed my main finding, and one of them reversed it back, so
they're worth reading before you trust any number in the README.

## What I asked for

I wrote the build brief. That covered the target and why (HDAC1 acts directly on chromatin, so
it's a cleaner example of epigenetic reprogramming than the signalling compounds that show up in
the same reprogramming cocktails), the seven build phases, the rule that each phase gets checked
by an agent that didn't build it, and what to do if the GNN stalled or the classes came out
badly imbalanced.

Then I made four calls during planning: scripts instead of one big notebook, tuning on a
validation fold with multi-seed reporting instead of fixed defaults, include the learning-curve
and similarity diagnostics, and build everything locally without pushing.

Later I asked for a harsh review of the repo as a work sample. It came back with four
criticisms. I acted on all four. The first one destroyed my main result.

## Four things that went wrong

### 1. A guard that never actually ran

`src/evaluate.py` was supposed to stop the correlation metrics from scoring a constant
prediction:

```python
if np.std(y_pred) == 0 or np.std(y_true) == 0:
    return float("nan")
```

A test threw a warning that shouldn't have been possible if that worked. It turns out `np.std`
of an array of identical floats isn't exactly zero. The mean usually isn't exactly representable
in floating point, so you get left with about 1e-15:

```
np.std(pred) == np.float64(1.7763568394002505e-15) | == 0 -> False
```

So the guard was dead code and had been the whole time. The test passed anyway because SciPy
noticed the constant input on its own and returned `NaN`, which is the answer I was checking
for.

Fixed with `np.ptp`, which is just max minus min and has no arithmetic in it, so it's exactly
zero when the values really are identical. This one mattered: a model that collapsed to
predicting the mean would have slipped through and been scored with a meaningless correlation.

### 2. A threshold I'd assumed instead of measured

I'd written into the plan that the median test-to-train Tanimoto should land between 0.30 and
0.45, and fail above 0.60. On the real split it came out at 0.66 and four gates failed.

The split wasn't broken. My threshold was. I measured a random split at the same fold sizes and
got 0.786, and the dataset's own nearest-neighbour median is about 0.80. HDAC1's ChEMBL data is
congeneric — nearly every compound is a zinc-binding group, a linker and a cap, pulled from
around 780 SAR papers about one pharmacophore. The 0.30–0.45 number describes diverse
multi-target benchmarks. Nothing scaffold-based is going to hit it on data like this.

The gate now compares against a measured random-split control instead of a number I made up.
The split data itself never changed — it hashes to the same `d25042a5…` before and after, which
I checked specifically because I didn't want to be the person who "fixed" a failing check by
moving the split.

I also stopped the agent from resolving this on its own and made the call myself. Picking a
split because it produces a nicer number is exactly the thing the gate was there to prevent.

While looking at this I found eight cross-fold pairs at Tanimoto 1.0. I checked all eight by
hand. They're genuinely different molecules that a folded ECFP4 can't tell apart — macrocycles
differing by one ring carbon, a sulfonamide with the N and S swapped, and linker homologs
differing by a single CH₂. So standardization had worked fine; the gate was using fingerprint
identity as a stand-in for structural identity, and the stand-in was wrong 8 times out of 8.
They're reported as flags now, and a separate check on canonical SMILES does the real job.

### 3. Adding the standard baseline killed my headline

The review's first point was that I'd never run Chemprop, which is the D-MPNN the field actually
benchmarks against. That leaves an obvious hole: maybe the graph side lost because *my* graph
model was weak, not because graph models struggle at this size.

At that point my headline was that random splits flatter fingerprint models more than twice as
much as graph models (+0.1285 versus +0.0560), enough to flip which model appears to win. I
added Chemprop expecting it to back that up. It came out at +0.1246, within 0.004 of the forest.
The asymmetry was a property of my weaker network, not of graph models.

I'd built the whole README around that claim.

### 4. The same bias twice more, by two different routes

It didn't stop there. Extending to three targets flipped the direction *again*, and both times
the cause was a bug I'd written.

**First bug.** The control trained the neural models on the random split using an epoch budget
that had been picked on the *scaffold* split. The random split is easier, so it supports longer
training — 249 epochs versus 142 for GINE. Capping them at the scaffold number under-trained
them on the easier split and made their optimism look smaller than it was.

**Second bug.** After fixing that, the control still computed optimism as an *ensembled*
scaffold RMSE minus a *single-seed* random RMSE. Seed-ensembling is worth about 0.04 RMSE to a
neural model and about 0.001 to a thousand-tree forest, which is already an ensemble. Ensembling
only one side of a subtraction takes 0.04 off the neural models and nothing off the forest.

The pattern is the same both times, and it's worth stating plainly: the forest has no epoch
schedule and gains nothing from seed averaging, so **any protocol detail I applied unevenly
landed entirely on the neural models.** Both bugs did. And both pushed the answer toward the
conclusion I'd already written down, which is the part I find uncomfortable.

Comparing like with like, the corrected numbers are forest +0.1289, GINE +0.1529, Chemprop
+0.1610. The graph models gain *more* from a random split, and that replicates on all three
targets.

Neither bug got caught by anyone reading the code. They got caught because two different scripts
computed the same quantity under different assumptions and disagreed, and the disagreement was
too big to wave off. `scripts/99_verify.py` now checks both traps directly, so they can't come
back quietly.

## What was mine and what wasn't

**Claude wrote effectively all of the code.** I didn't type these modules. Read that literally.

What was mine was the spec and the decisions:

- The brief, including the methodology — the verification-by-a-different-agent rule, the fallback
  clauses, and the reasoning for the target.
- The four planning calls.
- Stopping the agent when the gates failed and making the gate-versus-split call myself.
- Asking for the review that broke my own result, and deciding which of its criticisms to act on.

What that doesn't let me claim: I didn't derive the D-MPNN, write the bootstrap, or work out on
my own that `np.std` of identical floats isn't zero. If you're reading this as evidence that I
can engineer, weight it accordingly. The fair test is whether I can explain `src/evaluate.py`
and `src/scaffold_split.py`, so ask me about those.

A few things I did keep honest on purpose:

- The classification threshold was set at the conventional 100 nM cut before anything was scored,
  and I didn't move it afterwards.
- Failing gates got investigated rather than loosened.
- One of 27 model-pair × metric tests came out at p < 0.05, which is about what 27 tests give you
  by chance. The README says that instead of quoting the result.
- The verdict sentence in the README is generated from the confidence interval in
  `results/comparison.json`, so the wording can't drift away from the numbers.

Every issue described above is fixed in the current version, and `scripts/99_verify.py` passes.
The two flags it still reports are the ECFP4 bit collisions from section 2 — those are reported
on purpose, not left unfixed.
