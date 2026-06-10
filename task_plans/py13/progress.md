# Progress：Python 3.9 → 3.13 升級

## 階段狀態

| Phase | 內容 | 狀態 |
|-------|------|------|
| 0 | 現狀偵查 + 計劃建立 | ✅ complete（2026-06-11） |
| 1 | labber_io 測試定錨 + Labber 依賴移除 | ⬜ pending |
| 2 | 依賴 pin 解鎖（3.9 上） | ⬜ pending |
| 3 | 切換直譯器到 3.13 | ⬜ pending（依賴 D2 決策） |
| 4 | 全量驗證與修復 | ⬜ pending |
| 5 | typing 現代化（optional） | ⬜ pending（依賴 D4 決策） |

## 會話日誌

### 2026-06-11 — Session py13_main（計劃建立）

- 用戶開題：升 Python 3.13；原 3.9 鎖定原因 = Labber 依賴，已被自寫 `labber_io.py` 取代。
- 並行委派 2 個 Explore（sonnet）：
  - Labber 使用點 → **原始碼零 Labber import**，只剩 pyproject/uv.lock 宣告 + `datasaver.py:190` bare-import bug + labber_io 無測試。
  - 3.9 鎖定點 → hard blockers = numpy 1.23.5 / matplotlib 3.9 / design extra；Pyro4 cgi 需驗證；server extra 留板端不動。
- 寫入 task_plan.md / findings.md / progress.md 三件套。
- **下一步：等用戶定奪 D1–D4，然後啟動 Phase 1。**
