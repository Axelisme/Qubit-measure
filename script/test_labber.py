import os
import subprocess
from pathlib import Path

LABBER_API_DIR = Path(__file__).parent.parent / "labber_api"


def convert_labber_file(labber_file_path: Path) -> None:
    if not labber_file_path.exists():
        raise FileNotFoundError(
            f"Cannot find labber api directory: {LABBER_API_DIR}, please ensure it is exists"
        )

    subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            LABBER_API_DIR,
            os.path.join(LABBER_API_DIR, "convert_labber_file.py"),
            "-f",
            os.path.abspath(labber_file_path),
        ]
    )
