#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

from project_paths import DATA_CLEAN, DATA_RAW, VERSION, ensure_dirs


ADDICTION_CAUSES = {
    "Substance use disorders",
}

DEPRESSION_CAUSES = {
    "Depressive disorders",
}

SELFHARM_CAUSES = {
    "Self-harm",
}

GBD_FILES = {
    "IHME-GBD_2023_DATA-age-standardized-death-rate-1.csv": {
        "output": "gbd_addiction_clean.csv",
        "filter_causes": ADDICTION_CAUSES,
        "filter_measures": {"Deaths"},
        "filter_metrics": {"Rate"},
    },
    "IHME-GBD_2023_DATA-deaths-mental-substance-violence-1.csv": {
        "output": "gbd_selfharm_clean.csv",
        "filter_causes": SELFHARM_CAUSES,
        "filter_measures": {"Deaths"},
        "filter_metrics": {"Rate"},
    },
    "IHME-GBD_2023_DATA-dalys-causes-1.csv": {
        "output": "gbd_depression_dalys_clean.csv",
        "filter_causes": DEPRESSION_CAUSES,
        "filter_measures": {"DALYs (Disability-Adjusted Life Years)"},
        "filter_metrics": None,
    },
    "IHME-GBD_2023_DATA-probability-of-death-1.csv": {
        "output": "gbd_prob_death_clean.csv",
        "filter_causes": None,
        "filter_measures": None,
        "filter_metrics": None,
    },
    "IHME-GBD_2023_DATA-all-cause-burden-all-ages-1.csv": {
        "output": "gbd_allcauses_clean.csv",
        "filter_causes": None,
        "filter_measures": None,
        "filter_metrics": None,
    },
    "IHME-GBD_2023_DATA-risk-factor-burden-1.csv": {
        "output": "gbd_big_categories_clean.csv",
        "filter_causes": None,
        "filter_measures": None,
        "filter_metrics": None,
    },
}


def load_iso3_mapping(data_clean: Path) -> dict[str, str]:
    mapping_path = data_clean / "country_iso3_mapping.csv"
    if not mapping_path.exists():
        raise SystemExit(f"Missing {mapping_path}. Run 01_country_mapping.py first.")

    mapping: dict[str, str] = {}
    with mapping_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("source_type") == "GBD" and row.get("iso3"):
                if row.get("match_type") in {"lookup", "override"}:
                    mapping[row["name"]] = row["iso3"]
    return mapping


def insert_iso3_field(fieldnames: list[str]) -> list[str]:
    if "location_name" in fieldnames:
        idx = fieldnames.index("location_name") + 1
        return fieldnames[:idx] + ["iso3"] + fieldnames[idx:]
    return fieldnames + ["iso3"]


def clean_gbd_file(
    input_path: Path,
    output_path: Path,
    iso3_map: dict[str, str],
    filter_causes: set[str] | None,
    filter_measures: set[str] | None,
    filter_metrics: set[str] | None,
) -> int:
    row_count = 0
    with input_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            return 0
        output_fields = insert_iso3_field(reader.fieldnames)

        with output_path.open("w", newline="", encoding="utf-8") as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                if filter_causes is not None:
                    cause = row.get("cause_name", "")
                    if cause not in filter_causes:
                        continue
                if filter_measures is not None:
                    measure = row.get("measure_name", "")
                    if measure not in filter_measures:
                        continue
                if filter_metrics is not None:
                    metric = row.get("metric_name", "")
                    if metric not in filter_metrics:
                        continue

                location = row.get("location_name", "")
                row["iso3"] = iso3_map.get(location, "")
                writer.writerow(row)
                row_count += 1
    return row_count


def main() -> None:
    ensure_dirs()

    iso3_map = load_iso3_mapping(DATA_CLEAN)

    for filename, config in GBD_FILES.items():
        input_path = DATA_RAW / filename
        if not input_path.exists():
            print(f"Missing {input_path}, skipping.")
            continue
        output_path = DATA_CLEAN / config["output"]
        rows = clean_gbd_file(
            input_path,
            output_path,
            iso3_map,
            config["filter_causes"],
            config["filter_measures"],
            config["filter_metrics"],
        )
        print(f"[{VERSION}] Wrote {rows} rows to {output_path}")


if __name__ == "__main__":
    main()
