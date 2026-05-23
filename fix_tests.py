import os
import re


def fix_file(path):
    with open(path, "r") as f:
        content = f.read()

    # Add cast to imports if not present
    if "from typing import cast" not in content:
        content = content.replace(
            "from __future__ import annotations",
            "from __future__ import annotations\nfrom typing import cast",
        )

    # Replace val.fields["x"].value
    content = re.sub(
        r'([a-zA-Z0-9_]+(?:\.value)?\.fields\["[^"]+"\])\.value',
        r"cast(DirectValue, \1).value",
        content,
    )

    # Replace .is_unset
    content = re.sub(
        r'([a-zA-Z0-9_]+(?:\.value)?\.fields\["[^"]+"\])\.is_unset',
        r"cast(DirectValue, \1).is_unset",
        content,
    )

    # Replace .choices
    content = re.sub(
        r'([a-zA-Z0-9_]+(?:\.spec)?\.fields\["[^"]+"\])\.choices',
        r"cast(ScalarSpec, \1).choices",
        content,
    )

    # Replace .fields on values
    content = re.sub(
        r'([a-zA-Z0-9_]+(?:\.value)?\.fields\["[^"]+"\])\.fields',
        r"cast(CfgSectionValue, \1).fields",
        content,
    )

    # Replace .fields on specs
    content = re.sub(
        r'([a-zA-Z0-9_]+(?:\.spec)?\.fields\["[^"]+"\])\.fields',
        r"cast(CfgSectionSpec, \1).fields",
        content,
    )

    with open(path, "w") as f:
        f.write(content)


fix_file("tests/gui/test_cfg_schemas.py")
fix_file("tests/gui/ui/test_schema_overrides.py")
