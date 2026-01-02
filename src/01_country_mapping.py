#!/usr/bin/env python3
from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

from project_paths import DATA_CLEAN, DATA_RAW, VERSION, ensure_dirs

try:
    import pycountry
except ImportError as exc:
    raise SystemExit(
        "pycountry is required. Install dependencies with: pip install -r requirements.txt"
    ) from exc


OVERRIDES_RAW = {
    "bolivia (plurinational state of)": "BOL",
    "venezuela (bolivarian republic of)": "VEN",
    "iran (islamic republic of)": "IRN",
    "syrian arab republic": "SYR",
    "united republic of tanzania": "TZA",
    "cote d'ivoire": "CIV",
    "cote d ivoire": "CIV",
    "lao people's democratic republic": "LAO",
    "democratic people's republic of korea": "PRK",
    "republic of korea": "KOR",
    "republic of moldova": "MDA",
    "cabo verde": "CPV",
    "cape verde": "CPV",
    "timor-leste": "TLS",
    "viet nam": "VNM",
    "brunei darussalam": "BRN",
    "gambia": "GMB",
    "the gambia": "GMB",
    "bahamas": "BHS",
    "bahamas the": "BHS",
    "russian federation": "RUS",
    "united states of america": "USA",
    "united states": "USA",
    "turkey": "TUR",
    "turkiye": "TUR",
    "swaziland": "SWZ",
    "eswatini": "SWZ",
    "czech republic": "CZE",
    "czechia": "CZE",
    "north macedonia": "MKD",
    "micronesia (federated states of)": "FSM",
    "sao tome and principe": "STP",
    "holy see": "VAT",
    "state of palestine": "PSE",
    "palestine": "PSE",
    "democratic republic of the congo": "COD",
    "netherlands (kingdom of the)": "NLD",
    "mexico": "MEX",
    "republic of italy": "ITA",
    "republic of niue": "NIU",
    "republic of rwanda": "RWA",
    "republic of sudan": "SDN",
    "republic of turkey": "TUR",
    "republic of the union of myanmar": "MMR",
    "state of libya": "LBY",
    "taiwan (province of china)": "TWN",
    "united states virgin islands": "VIR",
    "state of eritrea": "ERI",
}


def normalize_key(name: str) -> str:
    cleaned = name.strip().lower()
    cleaned = unicodedata.normalize("NFKD", cleaned)
    cleaned = "".join(ch for ch in cleaned if not unicodedata.combining(ch))
    cleaned = cleaned.replace("&", "and")
    cleaned = re.sub(r"\(.*?\)", "", cleaned)
    cleaned = re.sub(r"[^a-z0-9\\s-]", " ", cleaned)
    cleaned = cleaned.replace("-", " ")
    cleaned = re.sub(r"\\s+", " ", cleaned)
    return cleaned.strip()


OVERRIDES = {normalize_key(k): v for k, v in OVERRIDES_RAW.items()}


def lookup_iso3(name: str, allow_fuzzy: bool = True) -> tuple[str | None, str]:
    if not name or not name.strip():
        return None, "empty"

    normalized = normalize_key(name)
    if normalized in OVERRIDES:
        return OVERRIDES[normalized], "override"

    candidates = [name.strip(), re.sub(r"\\s*\\(.*?\\)", "", name).strip()]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            country = pycountry.countries.lookup(candidate)
            return country.alpha_3, "lookup"
        except LookupError:
            pass

        if allow_fuzzy:
            try:
                matches = pycountry.countries.search_fuzzy(candidate)
                if len(matches) == 1:
                    return matches[0].alpha_3, "fuzzy"
            except LookupError:
                pass

    return None, "unmatched"


def collect_values(path: Path, column: str) -> set[str]:
    values: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if column not in (reader.fieldnames or []):
            return values
        for row in reader:
            value = row.get(column)
            if value:
                values.add(value.strip())
    return values


def main() -> None:
    ensure_dirs()

    who_names: set[str] = set()
    for path in sorted(DATA_RAW.glob("who_*csv")):
        who_names.update(collect_values(path, "country"))

    gbd_names: set[str] = set()
    for path in sorted(DATA_RAW.glob("IHME-GBD_2023_DATA-*.csv")):
        gbd_names.update(collect_values(path, "location_name"))

    mapping_rows = []
    unmatched_rows = []

    for source_type, names in [("WHO", who_names), ("GBD", gbd_names)]:
        for name in sorted(names):
            iso3, match_type = lookup_iso3(name, allow_fuzzy=(source_type == "WHO"))
            if iso3:
                mapping_rows.append(
                    {
                        "source_type": source_type,
                        "name": name,
                        "iso3": iso3,
                        "match_type": match_type,
                    }
                )
            else:
                unmatched_rows.append(
                    {"source_type": source_type, "name": name, "reason": match_type}
                )

    mapping_path = DATA_CLEAN / "country_iso3_mapping.csv"
    with mapping_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["source_type", "name", "iso3", "match_type"]
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    unmatched_path = DATA_CLEAN / "country_iso3_unmatched.csv"
    with unmatched_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source_type", "name", "reason"])
        writer.writeheader()
        writer.writerows(unmatched_rows)

    print(f"[{VERSION}] Wrote {len(mapping_rows)} matches to {mapping_path}")
    print(f"[{VERSION}] Wrote {len(unmatched_rows)} unmatched to {unmatched_path}")


if __name__ == "__main__":
    main()
