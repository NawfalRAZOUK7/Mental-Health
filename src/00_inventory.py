#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

from project_paths import DATA_CLEAN, DATA_RAW, REPO_ROOT, VERSION, ensure_dirs

TARGET_COLUMNS = [
    "year",
    "sex_name",
    "age_name",
    "cause_name",
    "metric_name",
    "measure_name",
]


def catalog_dataset(path: Path, repo_root: Path) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        available = [col for col in TARGET_COLUMNS if col in fieldnames]
        unique_values = {col: set() for col in available}
        row_count = 0

        for row in reader:
            row_count += 1
            for col in available:
                value = row.get(col)
                if value:
                    unique_values[col].add(value)

    def dump(values: set[str]) -> str:
        return json.dumps(sorted(values))

    record = {
        "dataset_path": str(path.relative_to(repo_root)),
        "row_count": row_count,
        "column_names": json.dumps(fieldnames),
    }

    for col in TARGET_COLUMNS:
        record[f"{col}_unique"] = dump(unique_values[col]) if col in unique_values else "[]"

    return record


def main() -> None:
    ensure_dirs()

    datasets = sorted(DATA_RAW.glob("*.csv"))
    catalog_rows = [catalog_dataset(path, REPO_ROOT) for path in datasets]

    output_path = DATA_CLEAN / "dataset_catalog.csv"
    fieldnames = [
        "dataset_path",
        "row_count",
        "column_names",
        "year_unique",
        "sex_name_unique",
        "age_name_unique",
        "cause_name_unique",
        "metric_name_unique",
        "measure_name_unique",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(catalog_rows)

    print(f"[{VERSION}] Wrote {len(catalog_rows)} datasets to {output_path}")


if __name__ == "__main__":
    main()
