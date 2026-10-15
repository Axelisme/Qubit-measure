"""Exercise parser serialization with controlled, in-process worker handoffs."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from matplotlib import _mathtext
from matplotlib.mathtext import MathTextParser
from zcu_tools.gui.plotting import install_mathtext_lock, prewarm_mathtext


@pytest.fixture(scope="module", autouse=True)
def parser_state_guard() -> Iterator[None]:
    original_parse = MathTextParser.parse
    assert hasattr(_mathtext.Parser, "parse"), "matplotlib Parser.parse seam is missing"
    original_effect = _mathtext.Parser.parse
    yield
    assert MathTextParser.parse is original_parse, "mathtext parser state leaked"
    assert _mathtext.Parser.parse is original_effect, "mathtext observer leaked"


def _install_with_restore(monkeypatch: pytest.MonkeyPatch, count: int) -> None:
    # Register restoration only: the public entry point is unchanged until install.
    monkeypatch.setattr(MathTextParser, "parse", MathTextParser.parse)
    for _ in range(count):
        install_mathtext_lock()
    prewarm_mathtext()
    # This observer depends on a matplotlib internal module path; recheck on upgrade.
    assert hasattr(_mathtext.Parser, "parse"), "matplotlib Parser.parse seam is missing"


@pytest.mark.parametrize("install_count", [1, 3], ids=["install", "reinstall"])
def test_mathtext_lock_serializes_concurrent_parsing(
    monkeypatch: pytest.MonkeyPatch, install_count: int
) -> None:
    _install_with_restore(monkeypatch, install_count)
    original_parse = _mathtext.Parser.parse
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    # Fresh expressions force both public calls through the real parser effect.
    token = uuid4().int
    expressions = (rf"$x_{{{token}}}+\alpha$", rf"$y_{{{token}}}+\beta$")
    records: list[tuple[str, str]] = []
    active: set[str] = set()
    overlaps: list[str] = []
    record_lock = threading.Lock()

    def observed_parse(
        parser: _mathtext.Parser,
        expression: str,
        fonts: _mathtext.Fonts,
        fontsize: float,
        dpi: float,
    ) -> _mathtext.Hlist:
        with record_lock:
            if active:
                overlaps.append(expression)
            active.add(expression)
            records.append(("enter", expression))
        try:
            if expression == expressions[0]:
                first_entered.set()
                # The caller's finally releases the first effect on every path.
                assert release_first.wait(timeout=30), "first parser was not released"
            elif expression == expressions[1]:
                second_entered.set()
            return original_parse(parser, expression, fonts, fontsize, dpi)
        finally:
            with record_lock:
                records.append(("exit", expression))
                active.remove(expression)

    monkeypatch.setattr(_mathtext.Parser, "parse", observed_parse)

    def parse(index: int):
        if index == 1:
            second_attempted.set()
        return MathTextParser("agg").parse(expressions[index])

    with ThreadPoolExecutor(max_workers=2) as pool:
        workers = [pool.submit(parse, 0)]
        try:
            # The first unique expression must enter the observed parser effect.
            assert first_entered.wait(timeout=30)
            workers.append(pool.submit(parse, 1))
            # The scheduled second worker reports its attempt, not queue arrival.
            assert second_attempted.wait(timeout=30)
            # No public lock-arrival seam: this window only adds detection power;
            # a correct lock cannot fail due to load or delayed scheduling.
            assert not second_entered.wait(timeout=0.05)
        finally:
            release_first.set()
            # Release lets both public parse calls finish; propagate worker errors.
            results = [worker.result(timeout=30) for worker in workers]

    assert second_entered.is_set()
    assert overlaps == []
    assert active == set()
    assert records == [
        ("enter", expressions[0]),
        ("exit", expressions[0]),
        ("enter", expressions[1]),
        ("exit", expressions[1]),
    ]
    assert len(results) == 2
    assert all(result.width > 0 for result in results)
