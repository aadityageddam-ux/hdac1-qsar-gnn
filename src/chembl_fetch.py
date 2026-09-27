"""Retrieve bioactivity records and assay metadata from the ChEMBL REST API.

Generic: takes a target id and filter parameters, and is not specific to any one
target. All HDAC1-specific values live in `scripts/`.

Data source: ChEMBL, https://www.ebi.ac.uk/chembl/ (unauthenticated public API).
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import asdict, dataclass

import pandas as pd
import requests

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
ACTIVITY_ENDPOINT = "activity"
ASSAY_ENDPOINT = "assay"
TARGET_ENDPOINT = "target"

# The REST API caps `limit` at 1000; anything larger is silently clamped.
PAGE_LIMIT = 1000

# `assay_chembl_id__in` is passed as a comma-joined list in the query string.
# 200 ids x ~14 characters keeps the URL comfortably under any proxy limit.
ASSAY_ID_CHUNK_SIZE = 200

DEFAULT_TIMEOUT = 120
MAX_RETRIES = 5
BACKOFF_CAP_SECONDS = 30

# 429 = EBI rate limiting; 500/502/503/504 = transient gateway or backend faults
# behind the EBI load balancer. Everything else (400 bad filter, 404 bad
# endpoint) is a programming error and is raised immediately.
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Columns kept from the activity endpoint. The endpoint returns ~45 fields per
# record; these are the ones any downstream QSAR step can actually use.
ACTIVITY_FIELDS: tuple[str, ...] = (
    "activity_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "canonical_smiles",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "data_validity_comment",
    "potential_duplicate",
    "assay_chembl_id",
    "assay_type",
    "assay_description",
    "document_chembl_id",
    "document_year",
    "target_chembl_id",
    "target_organism",
    "target_pref_name",
)

ASSAY_FIELDS: tuple[str, ...] = (
    "assay_chembl_id",
    "confidence_score",
    "assay_type",
    "assay_organism",
    "target_chembl_id",
)


class ChemblFetchError(RuntimeError):
    """Raised when the ChEMBL REST API cannot be queried or returns an unusable payload."""


@dataclass(frozen=True)
class FetchProvenance:
    """Immutable record of what was requested, when, and how much came back."""

    endpoint: str
    params: dict[str, str]
    total_count: int
    n_records: int
    n_pages: int
    retrieved_utc: str
    source_url: str

    def to_dict(self) -> dict:
        """Return the provenance record as a plain JSON-serialisable dict."""
        return asdict(self)


def _utc_now() -> str:
    """Return the current UTC time as a second-resolution ISO-8601 string."""
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _get_json(url: str, params: dict[str, object], timeout: int = DEFAULT_TIMEOUT) -> dict:
    """GET a ChEMBL JSON payload, retrying transient failures with exponential backoff."""
    last_error: str | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, BACKOFF_CAP_SECONDS))
            continue

        if response.status_code == 200:
            try:
                return response.json()
            except ValueError as exc:
                raise ChemblFetchError(
                    f"Non-JSON response from {response.url}: {exc}; body={response.text[:300]!r}"
                ) from exc

        if response.status_code in RETRYABLE_STATUS:
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
            time.sleep(min(2**attempt, BACKOFF_CAP_SECONDS))
            continue

        raise ChemblFetchError(
            f"HTTP {response.status_code} from {response.url}: {response.text[:500]}"
        )

    raise ChemblFetchError(f"{url} failed after {MAX_RETRIES} attempts: {last_error}")


def _page(
    endpoint: str,
    params: dict[str, object],
    record_key: str,
    limit: int = PAGE_LIMIT,
    progress: bool = True,
) -> tuple[list[dict], int, int]:
    """Walk every offset page of a ChEMBL collection endpoint and return all records."""
    url = f"{CHEMBL_BASE_URL}/{endpoint}"
    records: list[dict] = []
    offset = 0
    total_count = -1
    n_pages = 0

    while True:
        page_params = dict(params)
        page_params.update({"limit": limit, "offset": offset, "format": "json"})
        payload = _get_json(url, page_params)

        if record_key not in payload:
            raise ChemblFetchError(
                f"Payload from {endpoint} has no {record_key!r} key; got {sorted(payload)}"
            )
        meta = payload.get("page_meta", {})
        total_count = int(meta.get("total_count", -1))

        batch = payload[record_key]
        records.extend(batch)
        n_pages += 1
        if progress:
            print(f"    {endpoint}: {len(records)}/{total_count}", flush=True)

        if not batch or not meta.get("next"):
            break
        offset += limit

    return records, total_count, n_pages


def _select(records: list[dict], fields: tuple[str, ...]) -> pd.DataFrame:
    """Project raw ChEMBL records onto `fields`, preserving column order."""
    if not records:
        return pd.DataFrame(columns=list(fields))
    return pd.DataFrame([{key: record.get(key) for key in fields} for record in records])


def fetch_activities(
    target_chembl_id: str,
    standard_type: str,
    assay_type: str,
    standard_units: str,
    standard_relation: str | None = None,
    standard_relation_in: tuple[str, ...] | None = None,
    require_pchembl: bool = True,
    fields: tuple[str, ...] = ACTIVITY_FIELDS,
) -> tuple[pd.DataFrame, FetchProvenance]:
    """Fetch activity records for a target under the given measurement filters."""
    if standard_relation is not None and standard_relation_in is not None:
        raise ChemblFetchError(
            "Pass standard_relation or standard_relation_in, not both."
        )

    params: dict[str, object] = {
        "target_chembl_id": target_chembl_id,
        "standard_type": standard_type,
        "assay_type": assay_type,
        "standard_units": standard_units,
    }
    if standard_relation is not None:
        params["standard_relation"] = standard_relation
    if standard_relation_in is not None:
        params["standard_relation__in"] = ",".join(standard_relation_in)
    if require_pchembl:
        params["pchembl_value__isnull"] = "false"

    records, total_count, n_pages = _page(ACTIVITY_ENDPOINT, params, "activities")
    frame = _select(records, fields)

    provenance = FetchProvenance(
        endpoint=ACTIVITY_ENDPOINT,
        params={key: str(value) for key, value in params.items()},
        total_count=total_count,
        n_records=len(frame),
        n_pages=n_pages,
        retrieved_utc=_utc_now(),
        source_url=f"{CHEMBL_BASE_URL}/{ACTIVITY_ENDPOINT}",
    )
    if total_count >= 0 and len(frame) != total_count:
        raise ChemblFetchError(
            f"Pagination incomplete: collected {len(frame)} of {total_count} activities."
        )
    return frame, provenance


def fetch_assays(
    assay_chembl_ids: list[str],
    chunk_size: int = ASSAY_ID_CHUNK_SIZE,
    fields: tuple[str, ...] = ASSAY_FIELDS,
) -> tuple[pd.DataFrame, FetchProvenance]:
    """Fetch assay metadata (confidence score, organism) for a list of assay ids."""
    unique_ids = sorted(set(assay_chembl_ids))
    if not unique_ids:
        raise ChemblFetchError("fetch_assays called with an empty id list.")

    frames: list[pd.DataFrame] = []
    n_pages = 0
    for start in range(0, len(unique_ids), chunk_size):
        chunk = unique_ids[start : start + chunk_size]
        records, _, pages = _page(
            ASSAY_ENDPOINT,
            {"assay_chembl_id__in": ",".join(chunk)},
            "assays",
            progress=False,
        )
        n_pages += pages
        frames.append(_select(records, fields))
        print(
            f"    {ASSAY_ENDPOINT}: {min(start + chunk_size, len(unique_ids))}/{len(unique_ids)}",
            flush=True,
        )

    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset="assay_chembl_id")
    provenance = FetchProvenance(
        endpoint=ASSAY_ENDPOINT,
        params={"assay_chembl_id__in": f"<{len(unique_ids)} ids in chunks of {chunk_size}>"},
        total_count=len(unique_ids),
        n_records=len(frame),
        n_pages=n_pages,
        retrieved_utc=_utc_now(),
        source_url=f"{CHEMBL_BASE_URL}/{ASSAY_ENDPOINT}",
    )
    if len(frame) != len(unique_ids):
        raise ChemblFetchError(
            f"Assay fetch incomplete: {len(frame)} of {len(unique_ids)} assay ids returned."
        )
    return frame, provenance


def fetch_target(target_chembl_id: str) -> dict:
    """Fetch the target record so the pipeline can assert it is the intended protein."""
    payload = _get_json(
        f"{CHEMBL_BASE_URL}/{TARGET_ENDPOINT}/{target_chembl_id}",
        {"format": "json"},
    )
    if "target_chembl_id" not in payload:
        raise ChemblFetchError(f"Unexpected target payload for {target_chembl_id}: {sorted(payload)}")
    return payload
