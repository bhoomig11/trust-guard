"""Download public-domain US regulatory guidance into data/policies/public/.

These are works of the US federal government (FinCEN) and are not subject to
copyright. They are downloaded on demand rather than committed to the repo.

Usage:
    python scripts/download_public_docs.py [--out data/policies/public]
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

PUBLIC_DOCUMENTS: dict[str, str] = {
    "fincen_ctr_reference_guide.pdf": (
        "https://www.fincen.gov/sites/default/files/shared/CTRPamphlet.pdf"
    ),
    "fincen_sar_narrative_guidance.pdf": (
        "https://www.fincen.gov/sites/default/files/shared/sarnarrcompletguidfinal_112003.pdf"
    ),
}

USER_AGENT = "TrustGuard-docs-downloader/0.1"
TIMEOUT_SECONDS = 60


def download(url: str, destination: Path) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:  # noqa: S310
        payload = response.read()
    if not payload.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    destination.write_bytes(payload)
    return len(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/policies/public"))
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    failures = 0
    for name, url in PUBLIC_DOCUMENTS.items():
        path = args.out / name
        if path.exists() and not args.force:
            print(f"skip {path} (exists)")
            continue
        try:
            size = download(url, path)
            print(f"wrote {path} ({size // 1024} KB)")
        except Exception as exc:  # network errors should not hide other downloads
            failures += 1
            print(f"failed {name}: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
