"""Observe official GUI folder browsing and save an explicitly confirmed parent.

The local authenticated bridge forwards account configuration and reads only.
It never copies or deletes cloud data. There is no custom HTML or frontend.
"""

from __future__ import annotations

import hmac
import json
import threading
import urllib.parse
import webbrowser
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .rclone import GuiEndpoint, gui_session
from .settings import TransferSettings, save_settings, validate_remote_parent


def _parameters(path: str, body: bytes, content_type: str) -> dict[str, object]:
    parameters: dict[str, object] = {
        name: values[-1]
        for name, values in urllib.parse.parse_qs(
            urllib.parse.urlsplit(path).query, keep_blank_values=True
        ).items()
    }
    if body:
        if content_type.startswith("application/json"):
            value: object = json.loads(body)
            if not isinstance(value, dict):
                raise ValueError("API parameters must be an object.")
            parameters.update(value)
        elif content_type.startswith("application/x-www-form-urlencoded"):
            parameters.update(
                {
                    name: values[-1]
                    for name, values in urllib.parse.parse_qs(
                        body.decode(), keep_blank_values=True
                    ).items()
                }
            )
    return parameters


class SelectionBridge(ThreadingHTTPServer):
    """Setup-only proxy; records the last successful folder refresh for confirmation."""

    def __init__(self, endpoint: GuiEndpoint) -> None:
        self.endpoint = endpoint
        self.selection_lock = threading.Lock()
        self.sequence = 0
        self.candidate: str | None = None
        super().__init__(("127.0.0.1", 0), _SelectionHandler)

    def begin_browse(self) -> int:
        """Clear the candidate as navigation begins and return a request sequence."""
        with self.selection_lock:
            self.sequence += 1
            self.candidate = None
            return self.sequence

    def observe(self, parameters: dict[str, object], sequence: int) -> None:
        """Record a successful remote directory listing, ignoring stale requests."""
        filesystem = parameters.get("fs")
        folder = parameters.get("remote", "")
        if not isinstance(filesystem, str) or not isinstance(folder, str):
            return
        remote_name, separator, base = filesystem.partition(":")
        remotes = self.endpoint.call("config/listremotes").get("remotes")
        if not separator or not isinstance(remotes, list) or remote_name not in remotes:
            return
        combined = "/".join(
            part.strip("/") for part in (base, folder) if part.strip("/")
        )
        candidate = validate_remote_parent(remote_name + ":" + combined)
        with self.selection_lock:
            if sequence != self.sequence:
                return
            self.candidate = candidate
            print(
                f"\nSelected cloud parent: {candidate}\nVerify this location, then press Enter here to save it.",
                flush=True,
            )

    def confirmed_candidate(self) -> TransferSettings:
        """Snapshot the selected folder; fail after local, pending or failed navigation."""
        with self.selection_lock:
            candidate = self.candidate
        if candidate is None:
            raise ValueError(
                "Browse a cloud folder in Explorer first; no cloud parent is selected."
            )
        self.endpoint.call(
            "operations/list",
            {"fs": candidate, "remote": "", "opt": {"dirsOnly": True}},
        )
        return TransferSettings(candidate)


def _allowed_operation(operation: str, parameters: dict[str, object]) -> bool:
    if operation == "core/command":
        return parameters.get("command") == "version"
    return operation.startswith(
        ("config/", "options/", "job/", "fscache/", "core/")
    ) or operation in {
        "rc/noop",
        "rc/noopauth",
        "serve/list",
        "mount/listmounts",
        "operations/list",
        "operations/stat",
        "operations/about",
        "operations/fsinfo",
        "operations/size",
    }


class _SelectionHandler(BaseHTTPRequestHandler):
    server: SelectionBridge

    def log_message(self, format: str, *args: object) -> None:
        """Suppress access logs because request parameters can contain credentials."""

    def _send(
        self, status: int, body: bytes, content_type: str = "application/json"
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", self.server.endpoint.gui_origin)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        # Browsers cancel in-flight listings during navigation.
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        """Permit only the owned GUI origin to preflight authenticated API requests."""
        if self.headers.get("Origin") != self.server.endpoint.gui_origin:
            self._send(403, b'{"error":"Unexpected browser origin"}')
            return
        self._send(200, b"")

    def do_GET(self) -> None:
        """Forward an authenticated setup read request."""
        self._handle()

    def do_POST(self) -> None:
        """Forward account configuration or a setup read and observe folder listings."""
        self._handle()

    def _handle(self) -> None:
        endpoint = self.server.endpoint
        if not hmac.compare_digest(
            self.headers.get("Authorization", ""), endpoint.authorization
        ):
            self._send(401, b'{"error":"Authentication required"}')
            return
        if self.headers.get("Origin", endpoint.gui_origin) != endpoint.gui_origin:
            self._send(403, b'{"error":"Unexpected browser origin"}')
            return
        operation = urllib.parse.urlsplit(self.path).path.strip("/")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= 1024 * 1024:
                raise ValueError("Setup request is too large.")
            body = self.rfile.read(length)
            content_type = self.headers.get(
                "Content-Type", "application/x-www-form-urlencoded"
            )
            parameters = _parameters(self.path, body, content_type)
            if not _allowed_operation(operation, parameters):
                # The official GUI treats 401/403 as expired session credentials.
                self._send(
                    409,
                    b'{"error":"Setup only: browse your cloud parent, then press Enter in the terminal. Use the CLI for transfers."}',
                )
                return
            sequence = (
                self.server.begin_browse() if operation == "operations/list" else 0
            )
            response = endpoint.request(
                self.path, body, content_type=content_type, method=self.command
            )
            if sequence and response.status == 200:
                self.server.observe(parameters, sequence)
            self._send(response.status, response.body, response.content_type)
        except (ValueError, RuntimeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode())
        except OSError:
            self._send(502, b'{"error":"Cannot reach the local rclone setup server"}')


def run_setup(
    binary: Path,
    config_dir: Path,
    cache_dir: Path,
    settings_dir: Path,
    *,
    no_browser: bool,
) -> TransferSettings:
    """Use official GUI for OAuth and folder browsing; Enter confirms and saves.

    config_dir holds the per-user rclone.conf account; settings_dir receives
    this checkout's transfer.json. Existing transfer.json is preserved on
    cancellation/failure. Returns only after the selected remote parent has
    passed a fresh directory listing.
    """
    with (
        gui_session(binary, config_dir, cache_dir) as endpoint,
        SelectionBridge(endpoint) as bridge,
    ):
        worker = threading.Thread(target=bridge.serve_forever, daemon=True)
        worker.start()
        try:
            gui = urllib.parse.urlsplit(endpoint.login_url)
            query = urllib.parse.parse_qs(gui.query)
            query["url"] = [f"http://127.0.0.1:{bridge.server_port}/"]
            login_url = urllib.parse.urlunsplit(
                gui._replace(query=urllib.parse.urlencode(query, doseq=True))
            )
            print(
                "In Remotes, connect your account. In Explorer, open the cloud parent and click Refresh.",
                flush=True,
            )
            print(
                "Google Drive's shared OAuth client is retiring in 2026; use your project's client if required.",
                flush=True,
            )
            if no_browser or not webbrowser.open(login_url):
                print(
                    "Open this private setup link locally; do not share it:\n"
                    + login_url,
                    flush=True,
                )
            while True:
                try:
                    input(
                        "Click Refresh in the chosen cloud folder, verify the printed location, then press Enter to save (Ctrl+C cancels): "
                    )
                except EOFError as exc:
                    raise RuntimeError(
                        "Setup cancelled: no confirmation was received; no location was saved."
                    ) from exc
                try:
                    selection = bridge.confirmed_candidate()
                except ValueError as exc:
                    print(str(exc), flush=True)
                    continue
                save_settings(settings_dir, selection)
                print(
                    f"Saved cloud parent: {selection.remote_parent}. Future upload/download commands do not need the GUI.",
                    flush=True,
                )
                return selection
        finally:
            bridge.shutdown()
            worker.join(timeout=5)
