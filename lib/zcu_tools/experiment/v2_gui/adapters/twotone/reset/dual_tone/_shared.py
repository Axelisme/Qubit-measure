"""Shared writeback wiring for the dual-tone reset calibration chain."""

from __future__ import annotations

# field_md_map for the final 'reset_120' module: each tested_reset tone field is
# overwritten from its calibrated md key. All three dual-tone steps (freq / power /
# length) propose the same module, so the mapping lives here as the single source.
RESET_120_FIELD_MD_MAP: tuple[tuple[str, str], ...] = (
    ("pulse1_cfg.freq", "reset_f1"),
    ("pulse2_cfg.freq", "reset_f2"),
    ("pulse1_cfg.gain", "reset_gain1"),
    ("pulse2_cfg.gain", "reset_gain2"),
)
