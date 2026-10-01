# Data: licence and attribution

The files in this directory are derived from ChEMBL, not authored here.

**Source.** ChEMBL data is from <https://www.ebi.ac.uk/chembl>. Records were pulled from the
live ChEMBL REST API (`https://www.ebi.ac.uk/chembl/api/data`) on 2026-09-26; the exact query
predicates, per-stage record counts and retrieval timestamps are in `provenance.json`. The
ChEMBL *release* version was not recorded at fetch time, so it is not stated here — treat the
retrieval timestamp as the provenance anchor and re-fetch with
`uv run python scripts/01_fetch_chembl.py` to get a dated pull of your own.

**Licence.** ChEMBL data is released under the Creative Commons Attribution-ShareAlike 3.0
Unported licence (CC BY-SA 3.0): <https://creativecommons.org/licenses/by-sa/3.0/>. The derived
files here (`hdac1_clean.csv`, `hdac1_censored.csv`, `split_assignment.csv`) therefore carry
CC BY-SA 3.0, not the MIT licence that covers this repository's code.

ShareAlike means that if you redistribute these files or a dataset derived from them, you must
do so under CC BY-SA 3.0 and keep this attribution.

The code in `src/` and `scripts/`, the figures in `results/` and the text in `README.md` are
MIT-licensed (see `../LICENSE`).
