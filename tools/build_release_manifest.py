#!/usr/bin/env python3
"""Create SHA-256 manifests for the release and archived model code."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_rows() -> list[dict[str, str | int]]:
    excluded = {
        Path("provenance/file_manifest.csv"),
        Path("provenance/drive_release_manifest.csv"),
        Path("provenance/model_code_inventory.csv"),
    }
    rows = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file()):
        relative = path.relative_to(ROOT)
        if relative in excluded or any(part in {".git", "__pycache__"} for part in relative.parts):
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def model_rows(rows: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    result = []
    for row in rows:
        path = str(row["path"])
        if not path.endswith(".py"):
            continue
        if path.startswith("src/models/"):
            status = "canonical"
        elif path.startswith("archive/hprc_code_snapshot/"):
            status = "hprc_snapshot"
        elif path.startswith("archive/original_model_code/"):
            status = "historical_local"
        else:
            continue
        result.append({**row, "status": status})
    return result


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
    *,
    lineterminator: str = "\r\n",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            lineterminator=lineterminator,
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = file_rows()
    write_csv(
        ROOT / "provenance/file_manifest.csv",
        rows,
        ["path", "bytes", "sha256"],
        lineterminator="\n",
    )
    write_csv(
        ROOT / "provenance/model_code_inventory.csv",
        model_rows(rows),
        ["path", "bytes", "sha256", "status"],
    )
    print(f"Manifested {len(rows)} files.")


if __name__ == "__main__":
    main()
