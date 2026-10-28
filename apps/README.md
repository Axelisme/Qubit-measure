# GUI composition roots

**讀者：** operator、framework-dev、agent。啟動 GUI 並查閱 repo 層的 lab 與 framework 組裝責任。

`measure.py`、`fluxdep.py`、`dispersive.py` 各提供 `main(argv: list[str] | None = None) -> int`。
它們解析 CLI，從本套件位置定位 repo root，再呼叫唯一的 `zcu_lab.assembly.lab_definitions()` 取得 frozen lab 描述。
根以 `functools.partial` 綁定各 behavior 的 lab 與程序選項，交給
`launch_gui_runtime(spec, options, *, behavior_factory, logging_namespaces)`。
`logging_namespaces` 直接取自 lab。Launcher 先設定 logging，再呼叫 zero-arg factory，讓註冊與建構失敗寫入 session log。

Measure 根選擇 default、custom 或 clean startup，並傳入 project root。
`MeasureGuiBehavior` 從 lab 註冊 components、建立含 templates 的 experiment registry，並取得 reload policy 與 workflow attachments。
Fluxdep 與 dispersive 根將 CLI 以純轉換組成 `AnalysisSelection(project_root, entry, component)`。
各 analysis behavior 先註冊 lab components，再呼叫 `AnalysisTarget.open(selection)`。
未指定 entry 時保持未開啟；只有 component 沒有 entry 時報錯，不使用 project callback 或舊 `project` keyword。
資料根目錄可用 `--project-root` 改變，程式碼與 log 仍以 repo root 定位。
Control port、log 路徑與 startup 選擇屬於本次程序，不放進 lab 描述。

在 repo root 執行，下列指令的 `--help` 只解析 CLI，不啟動 GUI：

```bash
uv run --directory . --no-sync -- python -m apps.measure --help
uv run --directory . --no-sync -- python -m apps.fluxdep --help
uv run --directory . --no-sync -- python -m apps.dispersive --help
```

移除 `--help` 會開啟 GUI。先以 `uv sync --directory . --extra gui` 安裝工作站 profile。
`scripts/app/run_*_gui.py` 只委派相應 `main(sys.argv[1:])`，Windows `.bat` 直接使用 module 指令。
MCP 組合根在 [mcp_servers/](../mcp_servers/README.md)。Measure MCP 注入同一種 lab 描述與 GUI module 名稱；兩個 analysis MCP 只注入 module 名稱，由啟動的 GUI behavior 註冊 components。Bridge 不知道使用者腳本路徑。
Lab 欄位與各入口 Protocol 的 owner 見 [ADR-0072](../docs/adr/0072-module-boundaries.md#lab-定義與組合根)。
Framework 與 lab 不反向 import `apps` 或 `mcp_servers`。依賴由 `.importlinter` 的 `composition-roots-above-framework-and-lab` 約束，見 [ADR-0068](../docs/adr/0068-remote-transport.md)。

Startup 與資料根目錄的操作範例見 [scripts README](../scripts/README.md#量測-gui-啟動腳本)。
