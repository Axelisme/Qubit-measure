"""Thread-safety of matplotlib mathtext parsing under the process-wide lock (BUG-1).

matplotlib's mathtext parser is a class-level singleton (not thread-safe). The
GUIs parse ``$...$`` titles on off-main worker threads, so concurrent parses
corrupt the shared parser and raise non-deterministic ``ParseException`` (wrapped
as ``ValueError``).

Both checks run in a subprocess because the lock is process-wide and installed
by mutating ``MathTextParser.parse`` in place — once installed in the pytest
process it stays installed, so a "without lock" run needs a clean interpreter.
Each subprocess uses the Agg backend (no Qt), so it exercises the parser race
directly without the GUI plot host.

The two subprocesses share the same worker body and differ only in whether the
lock is installed: this proves both that the race is real (without the lock) and
that the fix eliminates it (with the lock).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Many threads, each parsing distinct $...$ strings after a barrier so they hit
# the singleton parser simultaneously. The strings use subscripts/fractions —
# the parse paths that corrupt under contention in the spike.
_WORKER_BODY = """
import threading

errors = []
N_THREADS = 12
N_ITERS = 60
barrier = threading.Barrier(N_THREADS)

def worker(idx):
    p = MathTextParser("path")
    barrier.wait()
    for i in range(N_ITERS):
        try:
            p.parse(rf"$x_{{{idx}}} + \\alpha_{i} = \\frac{{1}}{{2}}$")
        except Exception as exc:  # ParseException surfaces as ValueError
            errors.append(repr(exc))

threads = [threading.Thread(target=worker, args=(k,)) for k in range(N_THREADS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
"""


def _run_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_concurrent_mathtext_parse_races_without_lock() -> None:
    """Sanity check that the race is real: many threads parsing without the lock
    produce at least one ParseException. This guards against the test silently
    passing because the workload stopped being contended."""
    script = textwrap.dedent(
        """
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.mathtext import MathTextParser
        # No install_mathtext_lock(), no prewarm: reproduce the raw race.
        {worker}
        print("errors:", len(errors))
        assert errors, "expected the unlocked parser to race, got zero errors"
        print("raced")
        """
    ).format(worker=_WORKER_BODY)
    proc = _run_subprocess(script)
    assert proc.returncode == 0, proc.stderr
    assert "raced" in proc.stdout


def test_concurrent_mathtext_parse_clean_with_lock() -> None:
    """The fix: with the process-wide lock installed and the parser prewarmed,
    the same concurrent workload produces zero ParseExceptions."""
    script = textwrap.dedent(
        """
        import matplotlib
        matplotlib.use("Agg")
        from zcu_tools.gui.plotting import install_mathtext_lock, prewarm_mathtext
        install_mathtext_lock()
        prewarm_mathtext()
        from matplotlib.mathtext import MathTextParser
        {worker}
        print("errors:", len(errors))
        assert not errors, "locked parser still raced: " + repr(errors[:3])
        print("clean")
        """
    ).format(worker=_WORKER_BODY)
    proc = _run_subprocess(script)
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout


def test_install_mathtext_lock_is_idempotent() -> None:
    """Installing the lock more than once (all three GUI entry points install it)
    must not stack wrappers: re-install is a no-op once the sentinel is set."""
    script = textwrap.dedent(
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.mathtext as mathtext
        from zcu_tools.gui.plotting import install_mathtext_lock

        install_mathtext_lock()
        wrapped_once = mathtext.MathTextParser.parse
        install_mathtext_lock()
        install_mathtext_lock()
        # The function object is unchanged: no new wrapper was layered on top.
        assert mathtext.MathTextParser.parse is wrapped_once, "wrapper was re-stacked"
        print("idempotent")
        """
    )
    proc = _run_subprocess(script)
    assert proc.returncode == 0, proc.stderr
    assert "idempotent" in proc.stdout
