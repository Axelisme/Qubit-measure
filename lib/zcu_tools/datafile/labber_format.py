"""Labber 1.8.6 bookkeeping labels, compound dtypes and attribute conversion.

Writer, reader, grouped and streaming owners share these exact on-disk values.
String and attribute helpers convert borrowed values without performing file I/O.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np

LABBER_VERSION = "1.8.6"
STEP_INSTRUMENT = "Generic - GPIB: , Step channels at localhost"
LOG_INSTRUMENT = "Generic - GPIB: , Log channels at localhost"
# dummy step channel Labber inserts for trace/vector logs with no real step axis
API_STEP_NAME = "Step index API"

# ----------------------------------------------------------------------------
# h5py string / attribute helpers (mirror Labber's SR_HDF5)
# ----------------------------------------------------------------------------

_VLEN_STR = h5py.special_dtype(vlen=str)


def decode_labber_attribute(value: Any) -> Any:
    """Decode a borrowed HDF5 attribute using Labber's NoneValue convention.

    Bytes become UTF-8 text, string/object arrays become new lists and numpy
    booleans become bools. Other values pass through by identity. Invalid UTF-8
    propagates UnicodeDecodeError; this does not read or mutate an HDF5 object.
    """
    if isinstance(value, (bytes, str)):
        s = value.decode("utf-8") if isinstance(value, bytes) else value
        return None if s == "NoneValue" else s
    if isinstance(value, np.ndarray):
        if value.dtype.type is np.bytes_:
            return [el.decode("utf-8") for el in value]
        if value.dtype.type is np.object_:
            return [
                el.decode("utf-8") if isinstance(el, bytes) else str(el) for el in value
            ]
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def labber_string_array(values: Sequence[str]) -> np.ndarray:
    """Copy text values into a new array with Labber's HDF5 vlen string dtype.

    The input sequence is not mutated. Invalid values propagate numpy errors;
    the returned array is owned by the caller and performs no file I/O.
    """
    return np.array(list(values), dtype=_VLEN_STR)


# ----------------------------------------------------------------------------
# compound dtypes for the bookkeeping datasets (exact match to Labber 1.8.6)
# ----------------------------------------------------------------------------


def _enum_i2(mapping: dict[str, int]) -> np.dtype:
    return h5py.special_dtype(enum=(np.int16, mapping))


CHANNELS_DTYPE = np.dtype(
    [
        ("name", _VLEN_STR),
        ("instrument", _VLEN_STR),
        ("quantity", _VLEN_STR),
        ("unitPhys", _VLEN_STR),
        ("unitInstr", _VLEN_STR),
        ("gain", "<f8"),
        ("offset", "<f8"),
        ("amp", "<f8"),
        ("highLim", "<f8"),
        ("lowLim", "<f8"),
        ("outputChannel", _VLEN_STR),
        ("limit_action", _VLEN_STR),
        ("limit_run_script", "?"),
        ("limit_script", _VLEN_STR),
        ("use_log_interval", "?"),
        ("log_interval", "<f8"),
        ("limit_run_always", "?"),
    ]
)

INSTRUMENTS_DTYPE = np.dtype(
    [
        ("hardware", _VLEN_STR),
        ("version", _VLEN_STR),
        ("id", _VLEN_STR),
        ("model", _VLEN_STR),
        ("name", _VLEN_STR),
        (
            "interface",
            _enum_i2(
                {
                    "GPIB": 0,
                    "TCPIP": 1,
                    "USB": 2,
                    "PXI": 3,
                    "Serial": 4,
                    "VISA": 5,
                    "Other": 6,
                    "None": 7,
                }
            ),
        ),
        ("address", _VLEN_STR),
        ("server", _VLEN_STR),
        ("startup", _enum_i2({"Set config": 0, "Get config": 1, "Do nothing": 2})),
        ("lock", "?"),
        ("show_advanced", "?"),
        ("Timeout", "<f8"),
        ("Term. character", _VLEN_STR),
        ("Send end on write", "?"),
        ("Lock VISA resource", "?"),
        ("Suppress end bit termination on read", "?"),
        ("Use specific TCP port", "?"),
        ("TCP port", "<f8"),
        ("Use VICP protocol", "?"),
        ("Baud rate", "<f8"),
        ("Data bits", "<f8"),
        ("Stop bits", "<f8"),
        ("Parity", _VLEN_STR),
        ("GPIB board number", "<f8"),
        ("Send GPIB go to local at close", "?"),
        ("PXI chassis", "<f8"),
        ("Run in 32-bit mode", "?"),
    ]
)

LOG_LIST_DTYPE = np.dtype([("channel_name", _VLEN_STR)])

STEP_LIST_DTYPE = np.dtype(
    [
        ("channel_name", _VLEN_STR),
        ("step_unit", _enum_i2({"Instrument": 0, "Physical": 1})),
        ("wait_after", "<f8"),
        (
            "after_last",
            _enum_i2({"Goto first point": 0, "Stay at final": 1, "Goto value...": 2}),
        ),
        ("final_value", "<f8"),
        ("use_relations", "?"),
        ("equation", _VLEN_STR),
        ("show_advanced", "?"),
        ("sweep_mode", _enum_i2({"Off": 0, "Between points": 1, "Continuous": 2})),
        ("use_outside_sweep_rate", "?"),
        ("sweep_rate_outside", "<f8"),
        ("alternate_direction", "?"),
    ]
)

RELATION_PARAMETERS_DTYPE = np.dtype(
    [
        ("variable", _VLEN_STR),
        ("channel_name", _VLEN_STR),
        ("use_lookup", "?"),
    ]
)

STEP_ITEMS_DTYPE = np.dtype(
    [
        ("range_type", _enum_i2({"Single": 0, "Start - Stop": 1, "Center - Span": 2})),
        ("step_type", _enum_i2({"Fixed step": 0, "Fixed # of pts": 1})),
        ("single", "<f8"),
        ("start", "<f8"),
        ("stop", "<f8"),
        ("center", "<f8"),
        ("span", "<f8"),
        ("step", "<f8"),
        ("n_pts", "<i4"),
        (
            "interp",
            _enum_i2({"Linear": 0, "Log": 1, "Log, #/decade": 2, "Lorentzian": 3}),
        ),
        ("sweep_rate", "<f8"),
    ]
)

CHANNEL_NAMES_DTYPE = np.dtype([("name", _VLEN_STR), ("info", _VLEN_STR)])
