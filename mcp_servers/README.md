# mcp_servers

**讀者：** framework-dev、agent。啟動並維護注入實驗室定義的 MCP 組合根。

**Last updated:** 2026-10-13

這裡放 agent 使用的 repo 組合根，使用工作站 Python 3.13 GUI/MCP profile。
`measure.py` bootstrap `zcu_lab.components`，再把 `zcu_lab.recipes.RECIPES` 注入 measure stdio server。
`workflows.py` 注入 workflows bridge 的 repo root 與 GUI 啟動設定。GUI launcher 再注入 workflow 註冊清單、alias preset 與 source loader。
兩個入口的公開 `main()` 委派既有 server，不另解析 CLI 參數。

```bash
uv run --directory <worktree> --no-sync -- python mcp_servers/measure.py
uv run --directory <worktree> --no-sync -- python mcp_servers/workflows.py
```

Repo 的 `.mcp.json`、`.codex/config.toml` 與 `.pi/mcp.json` 指向這兩個入口。
啟動 stdio server 不自行連接儀器或啟動 GUI。Agent 必須明確呼叫 lifecycle tools；量測與硬體操作仍須取得當次授權。
EOF 只清理 MCP session 並斷線，不關閉 GUI 或硬體。公開工具與回覆由 [measure MCP](../lib/zcu_tools/mcp/measure/README.md) 及 [workflows 宿主](../lib/zcu_tools/gui/app/workflows/README.md) 說明。

使用者裁定 agent 組合根與使用者入口分開，讓 `scripts/` 只放使用者自己執行的入口。
組合根需要同時知道 lab 定義與 framework，不能放進其中任一方。
Fluxdep、dispersive 不需要 lab 注入，維持各自 lib 的 `server.py`。
Repo 品質檢查放在 [tools/](../tools/README.md)，使用者 GUI、板端與資料入口放在 [scripts/](../scripts/README.md)。
Agent 的 workflow 行為驗證使用 workflows GUI 的模擬模式，不另建獨立模擬腳本。
