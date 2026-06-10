# Progress：Python 3.9 → 3.13 升級

## 階段狀態

| Phase | 內容 | 狀態 |
|-------|------|------|
| 0 | 現狀偵查 + 計劃建立 | ✅ complete（2026-06-11） |
| 1 | labber_io 測試定錨 + Labber 依賴移除 | ✅ complete（2026-06-11） |
| 2 | 依賴 pin 解鎖（3.9 上）＝numpy 2 遷移 | ✅ complete（2026-06-11） |
| 3 | 切換直譯器到 3.13 | ✅ complete（2026-06-11） |
| 4 | 全量驗證與修復 | ✅ complete（2026-06-11） |
| 5 | typing 現代化 | ✅ complete（2026-06-11，D8=方案 A） |

**Phase 4 終值：pytest 2594 passed（0 fail/0 xfail）、pyright 0 errors 0 warnings、ruff clean、三 GUI（measure/fluxdep/dispersive）3.13 下 MCP launch+state_check smoke 全過。**

## 會話日誌

### 2026-06-11 — Session py13_main（計劃建立）

- 用戶開題：升 Python 3.13；原 3.9 鎖定原因 = Labber 依賴，已被自寫 `labber_io.py` 取代。
- 並行委派 2 個 Explore（sonnet）：
  - Labber 使用點 → **原始碼零 Labber import**，只剩 pyproject/uv.lock 宣告 + `datasaver.py:190` bare-import bug + labber_io 無測試。
  - 3.9 鎖定點 → hard blockers = numpy 1.23.5 / matplotlib 3.9 / design extra；Pyro4 cgi 需驗證；server extra 留板端不動。
- 寫入 task_plan.md / findings.md / progress.md 三件套。
- 用戶定錨 D1–D6 + branch 策略（py13 → 端到端測過 → main；可動所有 tracked 檔）。

### 2026-06-11 — Phase 1 完成（plan-item-implementer/sonnet）

- `datasaver.py:190` bare import 修為 relative；新增 `tests/utils/test_labber_io.py`（35 tests）+ `tests/utils/test_datasaver.py`（4 tests）；pyproject 移除 labber 依賴、`uv lock`（連帶移除 attrdict/future/msgpack/PyQt5 等 transitive）。
- 驗證：全套件 **2593 passed, 1 xfailed**；pyright 0；ruff clean。
- **發現既有 bug**：`labber_io._read_trace_log` 對 no-outer、等長、Nentries>1 的 traces reshape 會 ValueError——以 `xfail(strict=True)` 定錨，留待後續修（見 findings）。
- **下一步：Phase 2 依賴 pin 解鎖。**

### 2026-06-11 — Phase 2 完成（兩輪 plan-item-implementer/sonnet）

- pyproject：移除 server/design extras、client 的 numpy/matplotlib pin、死的 `[tool.uv.extra-build-dependencies]`；resolve 結果（3.9 下）：numpy **2.0.2**、matplotlib 3.9.4、pandas 2.3.3（numpy 2 ABI 配套升）；qiskit_metal/pyside2 從 lock 消失。
- 第一輪 agent 誤報「pyright 83 errors 是既有」（用 git stash 交叉驗證，但 stash 不會降 venv 的 numpy）——orchestrator 親自查核否決，第二輪委派修復。
- numpy 2 stubs 掀出的 83 個 pyright 錯全修（型別誠實化：float()/asarray/正確標註；僅 2 處窄 ignore：mpl secondary_xaxis stub、qutip 無 stubs）。
- 驗證：pyright 0/0、pytest 2593 passed + 1 xfailed、ruff clean。
- **下一步：Phase 3 切 3.13。**

### 2026-06-11 — Phase 3 完成（plan-item-implementer/sonnet）

- `.python-version`→3.13、`requires-python`→`>=3.13`、venv 重建（3.13.11）。版本：numpy 2.4.6 / matplotlib 3.10.9 / scipy 1.15.3 / pandas **3.0.3** / PyQt6 **6.11.0** / h5py 3.16.0 / numba 0.65.1 / qick 0.2.406 / Pyro4 4.82。3.9 backports（exceptiongroup/tomli/importlib-metadata…）自動移除。
- Pyro4 3.13 驗證 ✅（import/naming/Daemon/Proxy + zcu_tools.remote 封裝全過）；import smoke 10 子套件全過。
- 非 GUI pytest：1527 passed + 1 xfailed 全綠。
- **Phase 4 待辦清單**：(A) PyQt6 6.11 offscreen segfault ×2 測試（待驗證是否上游回歸→pin 6.10.2）；(B) tests/gui/services/test_background.py teardown 崩潰（缺 quiesce，3.13 GC 更積極暴露）；(C) pyright 36 errors（agent 稱既有，**不實**——Phase 2 收尾為 0，實為 3.13+新依賴 stubs 掀出）。
- 附帶觀察：qick parser.py 大量 SyntaxWarning（上游未加 r"" 的 regex），無功能影響。

### 2026-06-11 — Phase 4 完成（investigator/opus + 三路 implementer/sonnet 並行）

- **PyQt6 6.11 回歸假設被推翻**（6.10.2 對照同樣崩）：真因 = PyQt 對 slot 內未捕捉例外一律 abort + 兩個早於升級的 fixture 契約瑕疵 → 修 fixture（MagicMock→真 `FakeDeviceInfo`/`YOKOGS200Info`；過時 `get_persisted_left_panel_width` mock→`get_persisted_startup()`），順手修了同病第三測試；**不 pin PyQt6**。
- `test_background.py` 補 autouse quiesce fixture（比照 test_device.py 模式），連跑 3 次穩定。
- pyright 36→0（runner `A|B` alias 改 overload+移除 child_type 傳參、Pydantic 必填覆蓋欄位窄 ignore、numpy 2.4 收窄誠實化、qutip stubs 窄 ignore）。
- labber_io `_read_trace_log` reshape bug 修復（等長多 entry 無外軸 → 回 stacked `(Nentries, n)`，與 docstring 一致）；xfail 轉正。
- 文件：CLAUDE.md Python 版本描述更新（client 3.13 / 板端 3.8）；program AI_NOTE venv 路徑 3.9→3.13。
- 終值見上表。**剩餘：Phase 5 typing 現代化（用戶排後期）；commit/merge main 由用戶決定。**
- 註：用戶於本日 commit `dc74c1e0` 把整套文件（CLAUDE.md/AI_NOTE/docs/adr/task_plans）納入 git 追蹤——文件變更現在會進 diff/commit。

### 2026-06-11 — 升級 commit + Phase 5 細部規劃

- **commit `03d72f3e`**（py13 branch）：Phase 1–4 全部變更入庫（54 檔，+1024/−2838）。
- Phase 5 規劃完成（impl-detail-planner/opus 調查 + orchestrator 定稿，見 task_plan.md）：Step 0 修 Union introspection guards 是硬性前置；ruff 採常駐 extend-select UP006/007/035/037/045 + 板端檔 per-file-ignores（start_server.py、remote/pyro.py 維持 3.8 相容）；PEP 695 規則與 `from __future__ import annotations` 移除皆不在範圍。
- **待用戶定（D7）**：typing_extensions→typing 遷移（256 行）做不做。

### 2026-06-11 — Phase 5 Steps 0–3 完成（D7=遷回 typing + 移除依賴）

- **Step 0**（implementer/sonnet）：Union introspection guards 修復——`analyze_params.py` 加 `_UNION_ORIGINS={typing.Union, types.UnionType}`、`dispatch.py` 雙比對；先紅後綠補 8 個 PEP 604 測試（7 紅證明失配→修後全綠）；`sweep.py:17` 的 `unwrap_model_annotation` 經查不受影響。全套件 2602 passed。
- **Steps 1–3**（implementer/sonnet）：ruff 常駐 `extend-select UP006/007/035/037/045` + 板端兩檔 per-file-ignores；自動修 3056 處 + unsafe fixes + 手工 35 處（28 檔）；typing_extensions→typing/collections.abc 遷移；pyproject 移除 typing-extensions、uv lock/sync。改動 426 檔。
- **發現技術阻礙（D8 待用戶定）**：13 個檔案的 TypedDict 用 PEP 728 `closed=True`/`extra_items=`，3.13 stdlib 不支援，仍須 import typing_extensions——但依賴宣告已移除，形成「import 未宣告的 transitive dep」的脆弱狀態，必須二選一：(A) 恢復宣告 typing-extensions 為直接依賴；(B) 移除 13 檔的 PEP 728 參數（放寬 extra-key 靜態檢查）換取真正零依賴。
- orchestrator 終態查核：pyright 0/0、ruff 全清、pytest 2602 passed；板端兩檔零變動。
- 註：IDE language server 仍以舊 3.9 直譯器分析（誤報「`|` 需 3.10+」等），CLI pyright 為準；建議用戶重啟 IDE 的 Python language server。

### 2026-06-11 — Phase 5 收尾（D8=方案 A，用戶定）

- `typing-extensions` 恢復為直接依賴（pyproject 附註解：僅供 PEP 728 TypedDict `closed=True`/`extra_items=`）；uv lock/sync。
- 順手修 `remote/pyro.py`：`Any/Literal` 改自 stdlib `typing` import（3.8 即有，板端反而更乾淨）——typing_extensions 殘留現在嚴格限於 13 個 PEP 728 檔。
- ruff 常駐 UP 規則保留（用戶確認過理由：守住改寫成果 + per-file-ignores 護板端檔）。
- 終值：pyright 0/0、pytest 2602 passed、ruff 全清。**Phase 5 完成，py13 計劃全部閉合（Phase 1–5）；剩 merge main 由用戶執行。**
