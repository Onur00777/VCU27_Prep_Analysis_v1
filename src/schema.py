"""VCU sinyal semasini yukler."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config" / "vcu_signals_schema.yaml"
LOGS_DIR = ROOT / "logs_input"

CORNERS = ("fl", "fr", "rl", "rr")


@lru_cache(maxsize=1)
def load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("vcu_signals_schema.yaml bos veya gecersiz")
    return data


def vehicle() -> dict:
    return load_schema().get("vehicle", {})


def enums() -> dict:
    return load_schema().get("enums", {})


def signals() -> list[dict]:
    return list(load_schema().get("signals", []))


def firmware_header() -> str:
    cols = [s["csv"] for s in signals() if s.get("logged")]
    sep = load_schema().get("meta", {}).get("csv_separator", ";")
    return sep.join(cols)


def label(enum_name: str, value) -> str:
    table = enums().get(enum_name, {})
    if value is None:
        return "-"
    try:
        key = int(value)
    except (TypeError, ValueError):
        return str(value)
    text = table.get(key, table.get(str(key)))
    return text if text is not None else str(key)


def _as_int(value) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def bit_labels(enum_name: str, value) -> list[str]:
    table = enums().get(enum_name, {})
    try:
        bits = _as_int(value)
    except (TypeError, ValueError):
        return []
    found = []
    for mask, name in table.items():
        try:
            if bits & _as_int(mask):
                found.append(str(name))
        except (TypeError, ValueError):
            continue
    return found
