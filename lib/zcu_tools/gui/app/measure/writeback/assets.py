"""Measure's preview and selected-Apply policy for arb asset proposals."""

from typing import Literal

from zcu_tools.cfg_model import ModuleSpec, module_arb_keys
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.types import DocumentSetCapture
from zcu_tools.resources.context.library import LibrarySnapshot
from zcu_tools.resources.entry import writeback as concrete
from zcu_tools.resources.waveform_assets import ArbWaveformError

from .contracts import ArbAssetValues
from .items import ArbAssetWriteback


def arb_asset_current(
    item: ArbAssetWriteback, current: DocumentSetCapture
) -> Literal["absent", "identical", "different"]:
    """Compare samples in the current entry without modifying its asset files."""
    return concrete.arb_asset_update_status(
        current.capture_storage().database_path,
        concrete.ArbAssetWriteback(item.target_name, item.samples),
    )


def preview_arb_asset(
    item: ArbAssetWriteback, path: str, current: DocumentSetCapture | None
) -> ArbAssetValues:
    """Return preview facts for path, using None for status without an open entry."""
    status = None if current is None else arb_asset_current(item, current)
    samples = item.samples
    return ArbAssetValues(
        actual_path=path,
        current=status,
        proposed={
            "sample_count": int(samples.time.size),
            "duration_us": float(samples.time[-1]),
            "peak_abs": samples.peak_abs,
        },
        unit=None,
        warning=(
            f"Apply will fail: arb asset {item.target_name!r} has different samples."
            if status == "different"
            else None
        ),
    )


def require_arb_references(
    spec: ModuleSpec,
    library: LibrarySnapshot,
    selected_assets: set[str],
    *,
    session_id: str,
    target: str,
) -> None:
    """Reject reachable arb keys missing from the active entry and selected batch.

    Use the active library's typed child Specs and bound asset owner. Do not
    read samples, check unrelated library references or write anything.
    InvalidInputError names the proposal ID, target and missing key.
    """
    assets = library.validation_context().assets
    for key in sorted(module_arb_keys(spec, library) - selected_assets):
        if assets is not None:
            try:
                assets.path_for(key)
            except ArbWaveformError as exc:
                if exc.reason not in {"data_key_not_found", "invalid_data_key"}:
                    raise
            else:
                continue
        raise InvalidInputError(
            f"Module proposal {session_id!r} ({target}) requires arb asset {key!r}; "
            "publish it in the same draft or select an existing asset"
        )
