"""Timestamp evidence accepted by the migration's public analysis helper."""

import pytest

from migrations.storage_migration.analysis_files import proven_created_at


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-10-20T08:00Z",
        "2026-10-20T08:00:00Z",
        "2026-10-20T08:00:00.1Z",
        "2026-10-20T08:00:00.123456Z",
        "2026-10-20T16:00+08:00",
        "2026-10-20T16:00:00+08:00",
        "2026-10-20T01:30:00.123456-06:30",
        "2026-10-20T08:00:00+00:00",
    ],
)
def test_proven_timestamp_retains_original_spelling(timestamp: str) -> None:
    assert proven_created_at(timestamp) == timestamp


@pytest.mark.parametrize(
    "timestamp",
    [
        None,
        20261020,
        {"timestamp": "2026-10-20T08:00:00Z"},
        "",
        "2026-10-20",
        "20261020T08:00:00Z",
        "2026-W43-2T08:00:00Z",
        "2026W432T08:00:00Z",
        "2026-293T08:00:00Z",
        "2026-10-20t08:00:00Z",
        "2026-10-20X08:00:00Z",
        "2026-10-20 08:00:00Z",
        "2026-10-20T08:00:00",
        "2026-10-20T08:00:00z",
        "2026-02-30T08:00:00Z",
        "2026-10-20T24:00:00Z",
        "2026-10-20T08:60:00Z",
        "2026-10-20T08:00:60Z",
        "2026-10-20T080000Z",
        "2026-10-20T08Z",
        "2026-10-20T08:00.1Z",
        "2026-10-20T08:00:00,1Z",
        "2026-10-20T08:00:00.1234567Z",
        "2026-10-20T08:00:00+0800",
        "2026-10-20T08:00:00+08",
        "2026-10-20T08:00:00+08:00:00",
        "2026-10-20T08:00:00+08:60",
        "2026-10-20T08:00:00+24:00",
        "2026-10-20T08:00:00Z\n",
    ],
)
def test_unproven_timestamp_is_unknown(timestamp: object) -> None:
    assert proven_created_at(timestamp) is None
