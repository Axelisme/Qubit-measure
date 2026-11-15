"""Self-contained six-level close-carrier Engine input.

Fixed energy-basis matrices, not an environment builder or scratch dependency.
The 0.5 MHz, 1000 ns row historically exceeded 2e-3 at these guard coordinates.
"""

from dataclasses import replace

import numpy as np
from zcu_tools.simulate.evolution import EnvelopeSpec, SecularDrive, SecularModel

from .accuracy_cases import AccuracyCase, coherent_source, drive_track, timeline


def close_carrier_case() -> AccuracyCase:
    """Return a fixed six-level 1000 ns, 0.5 MHz-spaced two-tone input."""
    energies = np.array(
        [
            -2.636072250780856,
            1.4836431625758357,
            1.4944626355732293,
            4.605770090593419,
            7.957143608907118,
            8.732442845752729,
        ],
        dtype=np.float64,
    )
    rates = np.array(
        [
            [
                0.0,
                4.837159184270632e-06,
                1.7722038621851612e-31,
                0.0004311169602159639,
                5.657095022519504e-34,
                1.7856026872229553e-05,
            ],
            [
                2.461842723422967e-10,
                0.0,
                1.7272243605303591e-06,
                1.2743195592939048e-30,
                0.00029427061077803054,
                1.4023223922806006e-29,
            ],
            [
                8.788366740357976e-36,
                1.6829581755276865e-06,
                0.0,
                3.667161217611486e-05,
                1.0719315487289543e-29,
                0.00038026106300344557,
            ],
            [
                1.2233879767516407e-11,
                7.105219909304302e-34,
                2.0984789132174215e-08,
                0.0,
                0.0003261180959593417,
                2.416230598495379e-33,
            ],
            [
                5.163618620996854e-45,
                5.2776143345709494e-11,
                1.9730279820853175e-36,
                1.0489782490805987e-07,
                0.0,
                0.00016186896655414617,
            ],
            [
                2.5361405280898966e-17,
                3.913509817857487e-37,
                1.0891202881400032e-11,
                1.2093660246947256e-37,
                2.5187867077521576e-05,
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    x0 = np.array(
        [
            [
                2.2175782580605624e-33j,
                -0.055643117064282085j,
                1.0950588965329643e-14j,
                -0.5401183784725124j,
                -6.187114367720069e-16j,
                0.10992179774090642j,
            ],
            [
                0.05564311706428208j,
                3.474539942650638e-29j,
                -0.005473025284156954j,
                2.935685130874804e-14j,
                0.44623621528477275j,
                9.741264495737758e-14j,
            ],
            [
                -1.09505889653296e-14j,
                0.005473025284156943j,
                -1.2508191507286996e-29j,
                0.15748243228956138j,
                -8.516769503152788e-14j,
                0.5072620002108804j,
            ],
            [
                0.5401183784725123j,
                -2.935685130874802e-14j,
                -0.15748243228956152j,
                -2.992998119091313e-31j,
                -0.4696874955985651j,
                -1.2786442446383245e-15j,
            ],
            [
                6.187114367720078e-16j,
                -0.4462362152847728j,
                8.516769503152775e-14j,
                0.4696874955985652j,
                7.718803721058695e-32j,
                -0.30412046016292676j,
            ],
            [
                -0.10992179774090646j,
                -9.741264495737737e-14j,
                -0.5072620002108805j,
                1.2786442446383211e-15j,
                0.3041204601629267j,
                -5.469632701229783e-31j,
            ],
        ],
        dtype=np.complex128,
    )
    model = SecularModel(
        (0,),
        "close-carrier-six-level",
        energies,
        np.eye(6, dtype=np.complex128),
        (
            SecularDrive("Pump", x0, np.zeros_like(x0)),
            SecularDrive("Stokes", x0, np.zeros_like(x0)),
        ),
        rates,
        np.zeros((6, 6)),
    )
    batch = timeline(1, 1000.0)
    envelope = EnvelopeSpec("constant")
    batch = replace(
        batch,
        tracks=(
            drive_track(
                batch,
                envelope,
                port="Pump",
                frequency=4.119715413356692,
                gain=0.049990844167734846,
            ),
            drive_track(
                batch,
                envelope,
                port="Stokes",
                frequency=4.120215413356692,
                gain=0.00497466886406641,
            ),
        ),
    )
    return AccuracyCase(
        "close_carrier_six_level", coherent_source(model), batch, ("common_frame",)
    )
