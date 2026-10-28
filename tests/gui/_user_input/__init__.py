"""User-facing GUI test driver and real Qt input helpers.

Import from this package; implementation modules group observation, collection
input and nested-loop delivery without changing the component driver API.
"""

from .driver import ComponentDriver as ComponentDriver
from .input import key_click as key_click
from .input import mouse_click as mouse_click
from .input import type_text as type_text
