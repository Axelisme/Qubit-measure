# mcp_servers

**讀者：** framework-dev、agent。啟動並維護注入實驗室定義的 MCP 組合根。

**Last updated:** 2026-10-22

這裡放 agent 使用的 repo 組合根，使用工作站 Python 3.13 GUI/MCP profile。
`measure.py` bootstrap `zcu_lab.components`，再把 `zcu_lab.recipes.RECIPES` 注入 measure stdio server。
Workflow 工具也由 `measure.py` 的同一 stdio server 提供。Measure GUI 的組合根注入 workflow 註冊清單，MCP 不載入 lab workflow 或建立 Engine。
公開 `main()` 委派既有 server，不另解析 CLI 參數。

```bash
uv run --directory <worktree> --no-sync -- python mcp_servers/measure.py
```

Repo 的 `.mcp.json`、`.codex/config.toml` 與 `.pi/mcp.json` 使用 measure 入口，不另開 workflow server。
啟動 stdio server 不自行連接儀器或啟動 GUI。Agent 必須明確呼叫 lifecycle tools；量測與硬體操作仍須取得當次授權。
EOF 只清理 MCP session 並斷線，不關閉 GUI 或硬體。公開工具與回覆由 [measure MCP](../lib/zcu_tools/mcp/measure/README.md) 說明。

使用者裁定 agent 組合根與使用者入口分開，讓 `scripts/` 只放使用者自己執行的入口。
組合根需要同時知道 lab 定義與 framework，不能放進其中任一方。
Fluxdep、dispersive 不需要 lab 注入，維持各自 lib 的 `server.py`。
Repo 品質檢查放在 [tools/](../tools/README.md)，使用者 GUI、板端與資料入口放在 [scripts/](../scripts/README.md)。
Agent 的 workflow 行為驗證使用 measure 的 headless facet、RPC 與 MCP 合成案例，不啟動第二份 session。
