"""HTTP transport helpers for experiment data files."""

from __future__ import annotations

import os


def upload_to_server(filepath: str, server_ip: str, port: int) -> bool:
    """
    Upload a file to a remote server.

    Args:
        filepath: The path to the file to upload.
        server_ip: The IP address of the server.
        port: The port number of the server.

    Returns:
        True if upload succeeded, False otherwise.
    """
    import requests

    filepath = os.path.abspath(filepath)
    url = f"http://{server_ip}:{port}/upload"
    try:
        with open(filepath, "rb") as file:
            files = {"file": (filepath, file)}
            response = requests.post(url, files=files)
        print(response.text)
        return response.status_code == 200
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


def download_from_server(filepath: str, server_ip: str, port: int) -> None:
    """
    Download a file from a remote server.

    Args:
        filepath: The path where the downloaded file will be saved.
        server_ip: The IP address of the server.
        port: The port number of the server.
    """
    import requests

    url = f"http://{server_ip}:{port}/download"
    response = requests.post(url, json={"path": filepath})
    assert response.status_code == 200, f"Fail to download file: {response.text}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        file.write(response.content)

    print(f"Download file to {filepath}")
