"""Public FormModel specializations carrying handwritten logical-test inputs."""

from pydantic import create_model
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import FormModel
from zcu_tools.gui.form_adapter import Form


def specialize_input[C: ConfigBase](
    form: Form[C], handwritten: type[FormModel]
) -> type[FormModel]:
    """Extend the generated input using its public FormModel subclass contract."""
    return create_model("LogicalInput", __base__=(handwritten, form.model))
