# GUI composition roots

**讀者：** operator、framework-dev、agent。啟動 GUI 並查閱 repo 層的 lab 與 framework 組裝責任。

`measure.py`、`fluxdep.py`、`dispersive.py` 各提供 `main(argv: list[str] | None = None) -> int`。
它們解析 CLI，從本套件位置定位 repo root，注入 lab 定義，再呼叫 framework runtime。
Measure 選擇 default、custom 或 clean startup，注入 experiment catalogs 與 workflow 註冊清單。
Fluxdep 與 dispersive 註冊 lab components，交給各自的 analysis app 開啟 entry。
資料根目錄可用 `--project-root` 改變，程式碼與 log 仍以 repo root 定位。

在 repo root 執行，下列指令的 `--help` 只解析 CLI，不啟動 GUI：

```bash
uv run --directory . --no-sync -- python -m apps.measure --help
uv run --directory . --no-sync -- python -m apps.fluxdep --help
uv run --directory . --no-sync -- python -m apps.dispersive --help
```

移除 `--help` 會開啟 GUI。先以 `uv sync --directory . --extra gui` 安裝工作站 profile。
`scripts/app/run_*_gui.py` 只委派相應 `main(sys.argv[1:])`，Windows `.bat` 直接使用 module 指令。
MCP 組合根在 [mcp_servers/](../mcp_servers/README.md)，將 module 名稱注入 bridge，lib 不知道此套件或使用者腳本路徑。
Framework 與 lab 不反向 import `apps` 或 `mcp_servers`。依賴由 `.importlinter` 的 `composition-roots-above-framework-and-lab` 約束，見 [ADR-0068](../docs/adr/0068-remote-transport.md)。

Startup 與資料根目錄的操作範例見 [scripts README](../scripts/README.md#量測-gui-啟動腳本)。
