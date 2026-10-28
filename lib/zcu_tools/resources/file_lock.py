"""Lock-file naming for persisted experiment resources."""

import getpass
import os
import tempfile
from hashlib import sha256
from pathlib import Path


def lock_path_for(data_path: Path) -> Path:
    """Return a runtime lock path keyed by the resolved absolute data file path.

    data_path may be relative or name a missing file. Symlinks and ``..`` are
    resolved before hashing, so equivalent paths share a lock on this machine.
    The root is ``$XDG_RUNTIME_DIR/qubit-measure/locks`` when that variable names
    a directory, otherwise ``tempfile.gettempdir()/qubit-measure-<user>/locks``.
    This creates the root with mode 0o700 on demand, but does not create the data
    file or acquire the lock. OSError from path resolution, directory creation
    or user lookup propagates; user lookup may also raise KeyError.
    """
    resolved = data_path.resolve(strict=False)
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        root = Path(runtime) / "qubit-measure" / "locks"
    else:
        root = (
            Path(tempfile.gettempdir()) / f"qubit-measure-{getpass.getuser()}" / "locks"
        )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"{sha256(os.fsencode(resolved)).hexdigest()}.lock"
