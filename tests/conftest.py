import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB_PATH = os.path.join(_REPO_ROOT, "lib")
if _LIB_PATH not in sys.path:
    sys.path.insert(0, _LIB_PATH)


@pytest.fixture(autouse=True)
def _np_seed():
    np.random.seed(0)
