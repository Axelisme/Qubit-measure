from __future__ import annotations

from zcu_tools.experiment.v2.autofluxdep.env import FluxDepInfoTracker


def test_flux_dep_info_tracker_keeps_current_first_and_last_snapshots() -> None:
    tracker = FluxDepInfoTracker()

    tracker.start_step(flux_value=0.1, flux_idx=0, cur_m=2.0, m_ratio=1.0)
    tracker.update(t1=10.0, smooth_t1=10.0)
    tracker.start_step(flux_value=0.2, flux_idx=1, cur_m=3.0, m_ratio=1.5)

    assert tracker.current.flux_value == 0.2
    assert tracker.current.flux_idx == 1
    assert tracker.current.t1 is None
    assert tracker.first.flux_value == 0.1
    assert tracker.first.t1 == 10.0
    assert tracker.last.flux_value == 0.2
    assert tracker.last.t1 == 10.0


def test_flux_dep_info_tracker_deepcopies_mutable_values() -> None:
    tracker = FluxDepInfoTracker()
    tracker.start_step(flux_value=0.1, flux_idx=0, cur_m=2.0, m_ratio=1.0)
    readout = {"gain": [0.5]}

    tracker.update(opt_readout=readout)
    readout["gain"].append(0.6)

    assert tracker.current.opt_readout == {"gain": [0.5]}
    assert tracker.first.opt_readout == {"gain": [0.5]}
    assert tracker.last.opt_readout == {"gain": [0.5]}


def test_flux_dep_info_tracker_missing_required_field_fast_fails() -> None:
    tracker = FluxDepInfoTracker()
    tracker.start_step(flux_value=0.1, flux_idx=0, cur_m=2.0, m_ratio=1.0)

    try:
        tracker.require("predict_freq", task_name="QubitFreqTask")
    except ValueError as exc:
        assert "predict_freq" in str(exc)
        assert "QubitFreqTask" in str(exc)
    else:
        raise AssertionError("predict_freq should fast-fail when unset")


def test_flux_dep_info_tracker_rejects_unknown_field() -> None:
    tracker = FluxDepInfoTracker()

    try:
        tracker.update(not_a_field=1.0)
    except AttributeError as exc:
        assert "not_a_field" in str(exc)
    else:
        raise AssertionError("unknown FluxDepInfo field should fail")


def test_flux_dep_info_tracker_supports_smoothing_from_last_success() -> None:
    tracker = FluxDepInfoTracker()
    tracker.start_step(flux_value=0.1, flux_idx=0, cur_m=2.0, m_ratio=1.0)
    tracker.update(smooth_pi_product=100.0, lenrabi_success_idx=0)

    tracker.start_step(flux_value=0.3, flux_idx=2, cur_m=4.0, m_ratio=2.0)
    cur_pi_product = 200.0
    prev_pi_product = tracker.last_or("smooth_pi_product", cur_pi_product)
    num_step = max(1, tracker.flux_idx - tracker.last_or("lenrabi_success_idx", 0))
    weight = 0.7**num_step
    smooth_pi_product = (1 - weight) * cur_pi_product + weight * prev_pi_product
    tracker.update(
        smooth_pi_product=smooth_pi_product,
        lenrabi_success_idx=tracker.flux_idx,
    )

    assert num_step == 2
    assert tracker.current.smooth_pi_product == 151.0
    assert tracker.first.smooth_pi_product == 100.0
    assert tracker.last.smooth_pi_product == 151.0
