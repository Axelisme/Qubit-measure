# 任務計畫：AI_NOTE.md → README.md 改名遷移

**最後更新：** 2026-06-11（**遷移完成**：18 檔 git mv + 引用全清 + CLAUDE.md/SKILL×3/4 py 註解改寫；pytest 2602 綠、ruff 清；改動已 staged 待 commit。snr.py 3 個 pyright 錯與本遷移無關，來自用戶後續 commit 的依賴/mock-soc 變動，另案處理）

## 目標

文件集已入 git 追蹤（dc74c1e0）後，把各模組的 `AI_NOTE.md` 改名為 `README.md`，讓 GitHub/forge 瀏覽目錄時直接渲染模組 cheat-sheet。

## 盤點（2026-06-11）

- **18 個 tracked `AI_NOTE.md`**，全在子目錄（lib 16 + tests 1 + root 無——root 舊檔已在 6a64edbd 移除）。
- **零撞名**：18 個所在目錄皆無既有 README.md。root `README.md`（專案主 readme，pyproject `readme=` 引用）與 `docs/adr/README.md`（ADR 索引）不在遷移範圍、語義不變。
- 引用點：`CLAUDE.md`（`### AI_NOTE.md` 慣例段 + 散見）、orchestrate SKILL 三副本（.claude/.agent/.codex）、lib 4 個 py 檔註解（fluxdep/services/fit.py、main/services/remote/{dispatch,events,__init__}.py）、task_plans 歷史紀錄（gui/ir/py13）。

## 決策

- **D1 歷史紀錄不改**：task_plans 各 area 的 findings/progress 歷史段落中的「AI_NOTE」是當時事實，保留；只改「活的」指示性文件（CLAUDE.md、SKILL、程式碼註解）。
- **D2 命名歧義防護**：CLAUDE.md 慣例段改寫時明確界定「模組 README.md」（原 AI_NOTE 語義）vs root README.md（專案 readme）vs docs/adr/README.md（ADR 索引）。
- **D3 SKILL 三副本**：依既有慣例改 .claude 副本後整檔 cp 同步 .agent/.codex。

## 步驟

1. `git mv` 18 檔 AI_NOTE.md → README.md（保 history）。
2. 改名後檔案內部自我引用（如有「AI_NOTE」字樣）同步改。
3. CLAUDE.md 慣例段改寫（含 D2 界定）；AGENTS/GEMINI 為 1 行指標檔不動。
4. orchestrate SKILL 三副本更新（AI_NOTE 字樣 → 模組 README）。
5. lib 4 個 py 檔註解更新。
6. 驗證：`git ls-files | grep AI_NOTE` 零殘留；grep "AI_NOTE" 只剩 task_plans 歷史；pytest/pyright/ruff 快跑（理論上零影響）。
7. agent 記憶（MEMORY.md 等）同步——orchestrator 自理。
