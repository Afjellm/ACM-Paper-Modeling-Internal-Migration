from __future__ import annotations

import os


def parse_rows_per_age_group(value: str | None) -> int | None:
    if value is None:
        return None

    normalized = str(value).strip()
    if normalized.upper() == "NONE":
        return None

    rows_per_age_group = int(normalized)
    if rows_per_age_group <= 0:
        raise ValueError("rows_per_age_group must be a positive integer or NONE")

    return rows_per_age_group


def rows_per_age_group_label(rows_per_age_group: int | None) -> str:
    if rows_per_age_group is None:
        return "full"
    return f"rows_{rows_per_age_group}"


def parse_positive_int(value: str | None, default: int, name: str) -> int:
    if value is None or str(value).strip() == "":
        return default

    parsed = int(str(value).strip())
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return parsed


def get_rows_per_age_group_from_env() -> int | None:
    return parse_rows_per_age_group(os.getenv("ROWS_PER_AGE_GROUP", "NONE"))
