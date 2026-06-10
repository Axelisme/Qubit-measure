"""Compact channel-table rendering for a QICK soccfg.

QICK's own ``print(soccfg)`` (``QickConfig.description()``) expands every channel
into many lines.  ``describe_soc`` collapses the part that matters day-to-day into
two small tables (generators and readouts), one row per channel showing its type,
the physical converter port, the sample rate and the maximum pulse / buffer length.
"""

from __future__ import annotations

from typing import Any, Protocol


class SocCfgLike(Protocol):
    """Structural contract describe_soc needs: dict-style access to soccfg fields.

    Kept structural (not the concrete ``QickConfig``) so callers holding a thinner
    handle — e.g. the GUI's ``SocCfgProtocol`` — pass without a cast, and so the
    helper stays decoupled from QICK's import graph.
    """

    def __getitem__(self, key: str) -> Any: ...


# ZCU216 RF data-converter port numbering (Xilinx convention): DAC tiles are
# 228-231, ADC tiles are 224-227.  The port label QICK prints (e.g. "0_230") is
# f"{block}_{tile + offset}", where (tile, block) are the two digits of the
# converter name.  This layout is board-specific, so describe_soc only claims to
# know ZCU216 (the only board this project targets).
_DAC_TILE_OFFSET = 228
_ADC_TILE_OFFSET = 224


def describe_soc(soccfg: SocCfgLike) -> str:
    """Render a compact per-channel table of ``soccfg``.

    Raises NotImplementedError for non-ZCU216 boards, since the converter port
    label is board-specific.
    """
    board = soccfg["board"]
    if board != "ZCU216":
        raise NotImplementedError(
            f"describe_soc only knows ZCU216 port labels, got board={board!r}"
        )

    gens = soccfg["gens"]
    gen_rows = [
        [
            str(ch),
            gen["type"],
            _port_label(gen["dac"], _DAC_TILE_OFFSET),
            f"{soccfg['rf']['dacs'][gen['dac']]['fs']:.3f}",
            _gen_pulse_len(gen),
        ]
        for ch, gen in enumerate(gens)
    ]

    readouts = soccfg["readouts"]
    ro_rows = [
        [
            str(ch),
            ro["ro_type"],
            _port_label(ro["adc"], _ADC_TILE_OFFSET),
            f"{soccfg['rf']['adcs'][ro['adc']]['fs']:.3f}",
            _readout_buf_len(ro),
        ]
        for ch, ro in enumerate(readouts)
    ]

    blocks = [
        f"QICK running on {board}, software version {soccfg['sw_version']}\n"
        f"Firmware built {soccfg['fw_timestamp']}",
        f"Generators ({len(gens)})\n"
        + _format_table(
            ["ch", "type", "blk", "fs(Msps)", "max pulse len"], gen_rows, "rllrl"
        ),
        f"Readouts ({len(readouts)})\n"
        + _format_table(
            ["ch", "type", "blk", "fs(Msps)", "buf maxlen"], ro_rows, "rllrl"
        ),
    ]
    return "\n\n".join(blocks)


def _port_label(name: str, offset: int) -> str:
    """Convert a converter name like "20" into its ZCU216 port label "0_230"."""
    tile, block = (int(c) for c in name)
    return f"{block}_{tile + offset}"


def _gen_pulse_len(gen: dict) -> str:
    """Envelope-memory length of a generator, as "<samples> smp (<us> us)".

    Returns "-" for generators without an envelope memory (e.g. muxed/const gens).
    """
    if "maxlen" not in gen:
        return "-"
    maxlen = gen["maxlen"]
    us = maxlen / (gen["samps_per_clk"] * gen["f_fabric"])
    return f"{maxlen} smp ({us:.3f} us)"


def _readout_buf_len(ro: dict) -> str:
    """Decimated buffer length of a readout, as "<samples> smp (<us> us)"."""
    maxlen = ro["buf_maxlen"]
    us = maxlen / ro["f_output"]
    return f"{maxlen} smp ({us:.3f} us)"


def _format_table(headers: list[str], rows: list[list[str]], aligns: str) -> str:
    """Render rows of pre-stringified cells as an aligned, indented table.

    ``aligns`` is one char per column: "r" right-justified, anything else left.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        padded = [
            cell.rjust(w) if a == "r" else cell.ljust(w)
            for cell, w, a in zip(cells, widths, aligns)
        ]
        return ("  " + "  ".join(padded)).rstrip()

    return "\n".join(fmt(row) for row in [headers, *rows])
