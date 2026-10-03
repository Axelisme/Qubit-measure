"""Exercise parser serialization with controlled, in-process worker handoffs."""

from __future__ import annotations

import threading
from collections.abc import Iterator

import pytest
from matplotlib.mathtext import MathTextParser, RasterParse
from zcu_tools.gui.plotting import install_mathtext_lock, prewarm_mathtext


@pytest.fixture(scope="module", autouse=True)
def parser_state_guard() -> Iterator[None]:
    original_parse = MathTextParser.parse
    yield
    assert MathTextParser.parse is original_parse, "mathtext parser state leaked"


@pytest.mark.parametrize("install_count", [1, 3], ids=["install", "reinstall"])
def test_mathtext_lock_serializes_concurrent_parsing(
    monkeypatch: pytest.MonkeyPatch, install_count: int
) -> None:
    original_parse = MathTextParser.parse
    first_entered = threading.Event()
    release_first = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    expressions = (r"$x_{0}+\alpha$", r"$x_{1}+\beta$")
    results: list[RasterParse | None] = [None, None]

    def observed_parse(
        parser: MathTextParser[RasterParse], expression: str
    ) -> RasterParse:
        if expression == expressions[0]:
            first_entered.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("first parser was not released")
        elif expression == expressions[1]:
            second_entered.set()
        return original_parse(parser, expression)

    monkeypatch.setattr(MathTextParser, "parse", observed_parse)
    for _ in range(install_count):
        install_mathtext_lock()
    prewarm_mathtext()

    def parse(index: int) -> None:
        if index == 1:
            second_attempted.set()
        results[index] = MathTextParser("agg").parse(expressions[index])

    threads = [
        threading.Thread(target=parse, args=(index,), daemon=True) for index in range(2)
    ]
    threads[0].start()
    try:
        assert first_entered.wait(timeout=5)
        threads[1].start()
        assert second_attempted.wait(timeout=5)
        # The second parser must stay outside until the first one is released.
        assert not second_entered.wait(timeout=0.05)
    finally:
        release_first.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert second_entered.is_set()
    assert all(result is not None and result.width > 0 for result in results)
