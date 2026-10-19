"""Verified rclone installation and owned local GUI process lifecycle."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

VERSION = "1.75.2"
# From the official release's SHA256SUMS. Update the whole matrix together.
# https://github.com/rclone/rclone/releases/download/v1.75.2/SHA256SUMS
ARCHIVE_SHA256 = {
    "linux-amd64": "349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702",
    "linux-arm64": "7e1e8d69654941b7b7df84ee74c5f7fb09cce7d5496947fb7bf16b55ff427d10",
    "osx-amd64": "574e1f31cf1c464e5fb0325cb66fcbdeaf1efa1802bd9b17f13c256fd74b6561",
    "osx-arm64": "610ab7d48615135dfc60b763858b11a426a0d3d1160c1bb76fb0ae6011b02edd",
    "windows-amd64": "8a9db2bad1dc79a65b3348fd857069598e2aab7cf4d118b77b1df23055aff818",
    "windows-arm64": "7eb9844c9e5b723b9ef28821891caa17f6d9d1a375c49365cbe04114cec8dcd9",
}


def _target_platform() -> str:
    systems = {"Linux": "linux", "Darwin": "osx", "Windows": "windows"}
    machines = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    system = systems.get(platform.system())
    machine = machines.get(platform.machine().lower())
    if system is None or machine is None:
        raise RuntimeError(
            "Supported platforms: Windows/macOS/Linux on x86-64 or ARM64."
        )
    return f"{system}-{machine}"


def user_directories() -> tuple[Path, Path]:
    """Return platform-specific (configuration, cache) directories for this tool."""
    home = Path.home()
    if sys.platform == "win32":
        config = Path(os.environ.get("APPDATA") or home / "AppData/Roaming")
        cache = Path(os.environ.get("LOCALAPPDATA") or home / "AppData/Local") / "Cache"
    elif sys.platform == "darwin":
        config = home / "Library/Application Support"
        cache = home / "Library/Caches"
    else:
        config = Path(os.environ.get("XDG_CONFIG_HOME") or home / ".config")
        cache = Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    return (
        config / "qubit-measure/result-transfer",
        cache / "qubit-measure/result-transfer",
    )


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare_rclone(cache_dir: Path) -> Path:
    """Install/verify pinned rclone in cache_dir; raise on network or integrity failure."""
    target = _target_platform()
    archive_name = f"rclone-v{VERSION}-{target}"
    directory = cache_dir / archive_name
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"{archive_name}.zip"
    expected = ARCHIVE_SHA256[target]
    if archive.exists() and _sha256(archive) != expected:
        raise RuntimeError(
            f"Cached archive failed SHA-256 verification. Remove {archive} and retry."
        )
    if not archive.exists():
        url = f"https://github.com/rclone/rclone/releases/download/v{VERSION}/{archive.name}"
        print(f"Downloading rclone {VERSION} ({target})...", flush=True)
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            download = Path(staging) / archive.name
            with (
                urllib.request.urlopen(url, timeout=60) as response,
                download.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            if _sha256(download) != expected:
                raise RuntimeError(
                    "Downloaded archive failed SHA-256 verification; nothing was installed."
                )
            download.replace(archive)

    binary_name = "rclone.exe" if target.startswith("windows-") else "rclone"
    binary = directory / binary_name
    # Extract only the known executable, never arbitrary archive paths.
    with zipfile.ZipFile(archive) as bundle:
        content = bundle.read(f"{archive_name}/{binary_name}")
    if not binary.exists() or _sha256(binary) != hashlib.sha256(content).hexdigest():
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            candidate = Path(staging) / binary_name
            candidate.write_bytes(content)
            candidate.chmod(0o700)
            candidate.replace(binary)
    return binary


def _collect_login_url(stream: TextIO, ready: queue.Queue[str | None]) -> None:
    # Do not echo rclone output: the login URL contains session credentials.
    for line in stream:
        marker = "GUI available at "
        if marker in line:
            ready.put(line.split(marker, 1)[1].strip())
    ready.put(None)


def _check_readiness(login_url: str) -> None:
    gui = urllib.parse.urlsplit(login_url)
    params = urllib.parse.parse_qs(gui.query, strict_parsing=True)
    api_url = params["url"][0]
    api = urllib.parse.urlsplit(api_url)
    if (
        gui.scheme != "http"
        or api.scheme != "http"
        or gui.hostname != "127.0.0.1"
        or api.hostname != "127.0.0.1"
    ):
        raise RuntimeError("rclone did not advertise loopback GUI/API addresses.")
    credentials = f"{params['user'][0]}:{params['pass'][0]}".encode()
    authorization = "Basic " + base64.b64encode(credentials).decode("ascii")
    # Loopback credentials must never travel through a configured HTTP proxy.
    local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        api_url.rstrip("/") + "/rc/noop",
        data=b"{}",
        headers={"Authorization": authorization, "Content-Type": "application/json"},
    )
    with local_http.open(request, timeout=10) as response:
        if response.status != 200 or response.read().strip() != b"{}":
            raise RuntimeError("rclone API failed its authenticated readiness check.")
    with local_http.open(
        urllib.parse.urlunsplit(gui._replace(path="/", query="")), timeout=10
    ) as response:
        if response.status != 200 or b"<html" not in response.read().lower():
            raise RuntimeError("rclone did not serve the embedded Web GUI.")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


@dataclass(frozen=True)
class ApiResponse:
    """Raw local rclone response; body may contain account data and must not be logged."""

    status: int  # HTTP status from the owned rclone server.
    body: bytes  # Unmodified response payload.
    content_type: str  # Response MIME type.


@dataclass(frozen=True)
class GuiEndpoint:
    """Addresses and per-process credentials for an owned loopback GUI/API pair."""

    login_url: str  # Private browser login URL; contains session credentials.
    api_url: str  # Loopback URL of the underlying rclone API.
    authorization: str  # HTTP Basic header for this API session only.
    gui_origin: str  # Browser origin permitted by the underlying API.

    def request(
        self,
        path: str,
        body: bytes = b"{}",
        *,
        content_type: str = "application/json",
        method: str = "POST",
    ) -> ApiResponse:
        """Forward a relative API request locally; return HTTP errors as responses.

        Transport failures raise OSError. Never sends session credentials through
        an environment HTTP proxy. path must begin with a single slash.
        """
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("API path must start with a single slash.")
        request = urllib.request.Request(
            self.api_url.rstrip("/") + path,
            data=body if method != "GET" else None,
            method=method,
            headers={
                "Authorization": self.authorization,
                "Content-Type": content_type,
                "Origin": self.gui_origin,
            },
        )
        local_http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = local_http.open(request, timeout=300)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return ApiResponse(
                response.code,
                response.read(),
                response.headers.get("Content-Type", "application/json"),
            )

    def call(
        self, path: str, parameters: dict[str, object] | None = None
    ) -> dict[str, object]:
        """Call a JSON API; raise on HTTP failure or a non-object response."""
        response = self.request("/" + path, json.dumps(parameters or {}).encode())
        if response.status != 200:
            raise RuntimeError(f"rclone API {path} failed (HTTP {response.status}).")
        value: object = json.loads(response.body)
        if not isinstance(value, dict):
            raise ValueError("rclone returned a non-object API response.")
        return value


@contextmanager
def gui_session(
    binary: Path, config_dir: Path, cache_dir: Path
) -> Generator[GuiEndpoint]:
    """Start a verified local GUI; stop its owned child on every context exit.

    Yields only after authenticated API and HTML readiness checks. The caller
    opens the browser. Does not start any transfer or alter cloud files.
    """
    command = [
        str(binary),
        "gui",
        "--addr",
        "127.0.0.1:0",
        "--api-addr",
        "127.0.0.1:0",
        "--no-open-browser",
        "--config",
        str(config_dir / "rclone.conf"),
        "--cache-dir",
        str(cache_dir / "runtime"),
    ]
    with subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Could not capture rclone startup output.")
        ready: queue.Queue[str | None] = queue.Queue()
        reader = threading.Thread(
            target=_collect_login_url, args=(process.stdout, ready), daemon=True
        )
        reader.start()
        try:
            try:
                login_url = ready.get(timeout=30)
            except queue.Empty as exc:
                raise RuntimeError(
                    "rclone GUI startup timed out after 30 seconds."
                ) from exc
            if login_url is None:
                raise RuntimeError(
                    f"rclone exited before readiness (exit code {process.wait()})."
                )
            _check_readiness(login_url)
            gui = urllib.parse.urlsplit(login_url)
            params = urllib.parse.parse_qs(gui.query)
            credentials = f"{params['user'][0]}:{params['pass'][0]}".encode()
            yield GuiEndpoint(
                login_url,
                params["url"][0],
                "Basic " + base64.b64encode(credentials).decode("ascii"),
                f"{gui.scheme}://{gui.netloc}",
            )
        finally:
            _stop_process(process)
            reader.join(timeout=5)
