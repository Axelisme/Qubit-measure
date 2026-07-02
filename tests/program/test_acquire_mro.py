from qick.qick_asm import AcquireMixin

from zcu_tools.program.base.improve_acquire import (
    EarlyStopMixin,
    ImproveAcquireMixin,
    RoundHookMixin,
    SingleShotMixin,
    TrackerMixin,
    TypedAcquireMixin,
)
from zcu_tools.program.v2.base import MyProgramV2


def _classes_defining(cls: type, method_name: str) -> list[type]:
    return [base for base in cls.__mro__ if method_name in base.__dict__]


def test_improve_acquire_mro_order_is_stable():
    assert ImproveAcquireMixin.__mro__[:7] == (
        ImproveAcquireMixin,
        RoundHookMixin,
        TrackerMixin,
        SingleShotMixin,
        EarlyStopMixin,
        TypedAcquireMixin,
        AcquireMixin,
    )


def test_finish_round_resolution_chain_is_stable():
    assert _classes_defining(ImproveAcquireMixin, "finish_round")[:4] == [
        RoundHookMixin,
        TrackerMixin,
        EarlyStopMixin,
        AcquireMixin,
    ]


def test_my_program_v2_keeps_acquire_mixins_before_qick_acquire_base():
    mro = MyProgramV2.__mro__

    assert mro.index(ImproveAcquireMixin) < mro.index(AcquireMixin)
    assert mro.index(RoundHookMixin) < mro.index(TrackerMixin)
    assert mro.index(TrackerMixin) < mro.index(EarlyStopMixin)
