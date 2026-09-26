"""HTTP transport helpers for experiment data files."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _require_ok(response: Any, *, action: str, url: str) -> None:
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to {action} via {url}: HTTP {response.status_code}: "
            f"{response.text}"
        )


def upload_to_server(filepath: str, server_ip: str, port: int) -> None:
    """
    Upload a file to a remote server.

    Args:
        filepath: The path to the file to upload.
        server_ip: The IP address of the server.
        port: The port number of the server.

    Raises:
        RuntimeError: If the server rejects the upload.
    """
    import requests

    filepath = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/upload"
    with open(filepath, "rb") as file:
        files = {"file": (filepath, file)}
        response = requests.post(url, files=files)
    _require_ok(response, action="upload file", url=url)
    logger.info("Uploaded %s to %s", filepath, url)


def download_from_server(filepath: str, server_ip: str, port: int) -> None:
    """
    Download a file from a remote server.

    Args:
        filepath: The path where the downloaded file will be saved.
        server_ip: The IP address of the server.
        port: The port number of the server.
    """
    import requests

    request_path = filepath
    local_path = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/download"
    response = requests.post(url, json={"path": request_path})
    _require_ok(response, action="download file", url=url)

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as file:
        file.write(response.content)

    logger.info("Downloaded %s from %s", local_path, url)
