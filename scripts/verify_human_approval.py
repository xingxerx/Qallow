#!/usr/bin/env python3
"""CI guard ensuring a human has explicitly approved the latest changes."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn


CONFIG_PATH = Path("config/human_approval.json")
PLACEHOLDER_VALUES = {"", "CHANGE_ME", "PENDING", "TBD", "UNASSIGNED", "CODEX", "AI"}
MAX_DEFAULT_DAYS = 30


def _fail(message: str) -> NoReturn:
    """Print a consistent error message and exit."""
    print(f"[HUMAN-APPROVAL] {message}", file=sys.stderr)
    sys.exit(1)


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        _fail(f"Missing human approval record: {CONFIG_PATH}")
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive
        _fail(f"Failed to parse {CONFIG_PATH}: {exc}")


def _parse_datetime(raw: str) -> datetime:
    try:
        iso = raw.strip()
    except AttributeError as exc:
        _fail(f"Field 'approved_at' must be a string: {exc}")
    if iso.endswith("Z"):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        _fail(f"Field 'approved_at' must use ISO-8601 format: {exc}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> None:
    data = _load_config()

    required = {"approved", "approved_by", "approved_at"}
    missing = required.difference(data)
    if missing:
        _fail(f"Missing keys in {CONFIG_PATH}: {', '.join(sorted(missing))}")

    approved = data["approved"]
    if not isinstance(approved, bool):
        _fail("Field 'approved' must be a boolean")
    if not approved:
        _fail("Human approval flag is false — manual review required")

    approved_by = str(data.get("approved_by", "")).strip()
    if not approved_by:
        _fail("Field 'approved_by' must contain the approving human's name")
    if approved_by.upper() in PLACEHOLDER_VALUES or "TODO" in approved_by.upper():
        _fail("Field 'approved_by' contains a placeholder; replace with the human approver's name")

    approved_at = _parse_datetime(str(data.get("approved_at", "")))

    valid_for_days = data.get("valid_for_days", MAX_DEFAULT_DAYS)
    if isinstance(valid_for_days, str):
        if not valid_for_days.isdigit():
            _fail("Field 'valid_for_days' must be a positive integer")
        valid_for_days = int(valid_for_days)
    if not isinstance(valid_for_days, int) or valid_for_days <= 0:
        _fail("Field 'valid_for_days' must be a positive integer")

    now = datetime.now(timezone.utc)
    if approved_at > now:
        _fail("Human approval timestamp cannot be in the future")

    age_days = (now - approved_at).total_seconds() / 86400.0
    if age_days > valid_for_days:
        _fail(
            f"Human approval expired: {age_days:.1f} days old (limit {valid_for_days} days)"
        )

    scope = data.get("scope", "general")
    notes = data.get("notes", "")
    print(
        "Human approval verified for scope='{}' by {} (age {:.1f} days)".format(
            scope,
            approved_by,
            age_days,
        )
    )
    if notes:
        print(f"Notes: {notes}")


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
