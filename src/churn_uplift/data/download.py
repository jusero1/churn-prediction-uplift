"""Download the two real datasets and record their checksums.

The project has two components, each on its own real, public source:

- **Cell2Cell** (`jpacse/datasets-for-churn-telecom` on Kaggle) — real telecom
  churn data from the Teradata Center for CRM at Duke University. Drives the
  churn propensity model and the survival analysis (tenure = months in service).
- **Hillstrom / MineThatData** — a real randomized e-mail experiment (treatment
  = e-mail vs. control), the canonical dataset for uplift modeling. Downloaded
  directly from minethatdata.com.

Usage
-----
    python -m churn_uplift.data.download
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

KAGGLE_DATASET = "jpacse/datasets-for-churn-telecom"
HILLSTROM_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_FILENAME = "hillstrom.csv"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def raw_dir() -> Path:
    path = project_root() / "data" / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_checksum(path: Path) -> None:
    """Append ``<sha256>  <filename>`` to ``data/checksums.txt`` (idempotent)."""
    checksums = project_root() / "data" / "checksums.txt"
    existing = checksums.read_text() if checksums.exists() else ""
    kept = [ln for ln in existing.splitlines() if not ln.endswith(path.name)]
    line = f"{sha256(path)}  {path.name}"
    checksums.write_text("\n".join([*kept, line]).strip() + "\n")


def download_cell2cell(target_dir: Path) -> None:
    """Download and unzip the Cell2Cell churn dataset via the Kaggle API."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(target_dir), unzip=True)


def download_hillstrom(target: Path) -> None:
    """Download the Hillstrom randomized-experiment CSV from minethatdata.com."""
    import urllib.request

    urllib.request.urlretrieve(HILLSTROM_URL, target)  # noqa: S310 - fixed http URL


def main() -> None:
    raw = raw_dir()
    force = os.environ.get("FORCE_DOWNLOAD") == "1"

    # Cell2Cell (churn + survival). The archive ships a couple of CSVs; the
    # training file is the one used downstream.
    if force or not any(raw.glob("*Train*.csv")):
        print(f"Downloading {KAGGLE_DATASET} ...")
        download_cell2cell(raw)
    for csv in raw.glob("*.csv"):
        if "hillstrom" not in csv.name.lower():
            record_checksum(csv)

    # Hillstrom (uplift).
    hillstrom = raw / HILLSTROM_FILENAME
    if force or not hillstrom.exists():
        print("Downloading Hillstrom MineThatData e-mail experiment ...")
        download_hillstrom(hillstrom)
    record_checksum(hillstrom)

    print("Ready. Files in data/raw/, checksums in data/checksums.txt:")
    for csv in sorted(raw.glob("*.csv")):
        print(f"  {csv.name}")


if __name__ == "__main__":
    main()
