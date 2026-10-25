# mcp_servers

**讀者：** framework-dev、agent。啟動並維護注入實驗室定義與 GUI module 的 MCP 組合根。

**Last updated:** 2026-10-25

這裡放 agent 使用的 repo 組合根，使用工作站 Python 3.13 GUI/MCP profile。
`measure.py` bootstrap `zcu_lab.components`，再把 `zcu_lab.recipes.RECIPES` 與 `gui_module="apps.measure"` 注入 measure stdio server。
Workflow 工具由同一 measure server 提供。GUI 的組合根注入 workflow 註冊清單，MCP 不載入 lab workflow 或建立 Engine。
`fluxdep.py` 與 `dispersive.py` 分別將 `apps.fluxdep`、`apps.dispersive` 注入對應 lib server，GUI module 自行註冊 lab components。
公開 `main()` 委派 lib server，不另解析 CLI 參數。

```bash
uv run --directory <worktree> --no-sync -- python mcp_servers/measure.py
uv run --directory <worktree> --no-sync -- python mcp_servers/fluxdep.py
uv run --directory <worktree> --no-sync -- python mcp_servers/dispersive.py
```

Repo 的 `.mcp.json`、`.codex/config.toml` 與 `.pi/mcp.json` 使用這三個入口，不另開 workflow server。
啟動 stdio server 不自行連接儀器或啟動 GUI。Agent 必須明確呼叫 lifecycle tools；量測與硬體操作仍須取得當次授權。
Bridge 在 repo root 以目前 interpreter 執行 `python -m apps.<app>`，不讀取 `scripts/` 路徑。
EOF 只清理 MCP session 並斷線，不關閉 GUI 或硬體。工具與回覆見 [measure MCP](../lib/zcu_tools/mcp/measure/README.md) 與 [fluxdep MCP](../lib/zcu_tools/mcp/fluxdep/README.md)。

`apps/` 與 `mcp_servers/` 同層，可 import lab 定義與 framework，兩者互不 import。
Framework 與 lab 不反向 import 組合根，由 `.importlinter` 的 `composition-roots-above-framework-and-lab` 保護，依據見 [ADR-0068](../docs/adr/0068-remote-transport.md)。
使用者 GUI 薄殼、板端與資料入口放在 [scripts/](../scripts/README.md)，品質檢查放在 [tools/](../tools/README.md)。
Agent 的 workflow 行為驗證使用 measure 的 headless facet、RPC 與 MCP 合成案例，不啟動第二份 session。
