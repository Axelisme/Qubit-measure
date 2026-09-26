# ZCU-Tools

ZCU-Tools 是 ZCU216/QICK 的量子量測工具集。工作站執行量測 GUI、Notebook、分析與模擬；ZCU 板端提供 QICK Pyro server。連接硬體前，先確認板端環境與資產版本。板端部署和相容性尚未在硬體上核實。

## 執行環境

- 工作站預設 Python 3.13，供 GUI、量測 runtime 與 MCP 使用；`design` / `quantum-metal` / Ansys stack 使用 Python 3.12。
- ZCU 板端獨立使用 PYNQ Python 3.8，不共用工作站的 uv 環境。

工作站在目標 worktree 安裝所需 profile；例如量測 GUI 使用 `uv sync --directory <worktree> --extra gui`。`client` extra 會從 upstream Git 安裝 QICK。其他環境要求依 [repo 操作指引](CLAUDE.md) 和 [腳本入口](scripts/README.md) 確認。

```bash
uv run --directory <worktree> --no-sync -- python scripts/run_measure_gui.py
```

板端以其 Python 3.8 啟動 [`scripts/start_server.py`](scripts/README.md#gui-與板端-server) 或 `start_server.ipynb`；需要 repo-root 的 [`bitfiles/`](bitfiles/README.md) 與 `lib/`，不能用上述工作站命令代跑。其餘 GUI、資料處理與模擬入口，以及網路和資料寫入副作用，見 [scripts/README.md](scripts/README.md)。

## 從哪裡讀起

- [Notebook 用途入口](notebook_md/README.md) 指向量測、分析與電路設計主題；執行流程由各 Notebook 說明。
- [領域詞彙](docs/CONTEXT.md) 與 [量測程式](lib/zcu_tools/experiment/v2/README.md)與[GUI](lib/zcu_tools/gui/README.md)的 README 提供實作定位；跨模組決策見 [ADR](docs/adr/README.md)。
- [測試目錄與 fixture](tests/README.md) 說明案例歸屬；[程式碼品質](docs/code-quality.md) 說明 review 判準。
- [品質工具](tools/README.md) 說明 gate、ratchet 與報表的執行及判讀。開發操作與環境仍依 [repo 操作指引](CLAUDE.md)。
