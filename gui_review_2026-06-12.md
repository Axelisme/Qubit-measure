# GUI 模組實作 Review 報告

**Last updated:** 2026-06-12  
**範圍:** `lib/zcu_tools/gui/app/main/`、`lib/zcu_tools/experiment/v2_gui/` 最近 GUI 相關實作，重點檢查 post-analysis、`singleshot/ge` adapter、remote/MCP surface 與 async operation lifecycle。

## 摘要

最近的 GUI 實作在 happy path 上已有相當測試覆蓋，尤其是 post-analysis UI/service 與 `singleshot/ge` adapter 映射。但 review 發現幾個重要邊界尚未被鎖住：

- `AnalyzeService` / `PostAnalyzeService` 的 operation token 管理不支援跨 tab 並發，但目前也沒有全域 gate 阻止跨 tab 並發。
- post-analysis 只接上 Qt UI，尚未接上 remote/MCP agent path。
- post-analysis 缺 adapter capability gate，非 opt-in adapter 會進 worker 後才失敗。
- analyze/post-analyze terminal slot 若在 service-side post-processing 丟例外，可能留下 busy tab 與 live operation handle。

## Findings

### High: Analyze / Post-Analyze operation token 會被跨 tab 並發覆寫

**位置**

- `lib/zcu_tools/gui/app/main/services/analyze.py:51`
- `lib/zcu_tools/gui/app/main/services/analyze.py:69`
- `lib/zcu_tools/gui/app/main/services/post_analyze.py:51`
- `lib/zcu_tools/gui/app/main/services/post_analyze.py:71`

**問題**

`AnalyzeService` 與 `PostAnalyzeService` 都用單一 `_active_token: int | None` 保存 live operation。啟動時只檢查 `self._state.is_tab_busy(tab_id)`，這只能阻止同一 tab 重入，不能阻止另一個 tab 同時 analyze / post-analyze。

若 tab A 與 tab B 同時 analyze：

1. tab A 建 token 1，`_active_token = 1`
2. tab B 建 token 2，`_active_token = 2`
3. tab A 先完成，terminal path 讀到 token 2 並 settle token 2
4. tab B 完成時 `_active_token` 已清空，token 1 可能永遠 pending，或 terminal path 狀態錯亂

**影響**

- `gui_analyze_wait` / `gui_analyze_poll` 可能等待錯 operation。
- `active_operation_count()` 可能維持非零，shutdown 以為仍有工作。
- tab busy state 可能與 operation handle 不一致。

**建議**

二選一：

- 若設計不允許任何 concurrent analyze：新增 service-level global gate，所有 analyze/post-analyze 共用明確的 active token，啟動第二個 operation 時 fail-fast。
- 若允許跨 tab analyze：把 token 改成 `dict[str, int]` keyed by `tab_id` 或 operation id，terminal callback 必須使用啟動時捕捉的 token，不讀全域 `_active_token`。

建議同時補測：

- 兩個 tab 同時 `AnalyzeService.start_analyze()`，確認第二個被拒或兩者各自 settle 正確 token。
- 同樣覆蓋 `PostAnalyzeService`。

### High: Post-analysis 沒有 remote/MCP surface

**位置**

- `lib/zcu_tools/gui/app/main/services/remote/method_specs.py:462`
- `lib/zcu_tools/gui/app/main/services/remote/dispatch.py:1661`
- `lib/zcu_tools/mcp/measure/server.py:964`

**問題**

post-analysis 目前只接上 Qt UI：

- UI button 呼叫 `Controller.start_post_analyze()`
- `PostAnalyzeService` 執行 worker
- `MainWindow` 刷新 post figure

但 remote/MCP layer 沒有對應方法：

- 沒有 `adapter.post_analyze_spec`
- 沒有 `tab.get_post_analyze_params`
- 沒有 `post_analyze.start`
- 沒有 `tab.get_post_analyze_result`
- MCP 沒有 `gui_post_analyze` / wait / poll tool

**影響**

measure-gui 的 agent client 不能使用這個新功能。這和目前架構中「Qt UI 與 RemoteControlAdapter 是平級 client，共用 Controller façade」的設計不一致。

**建議**

補齊 wire/MCP surface：

- RPC:
  - `adapter.post_analyze_spec`
  - `tab.get_post_analyze_params`
  - `post_analyze.start`
  - `tab.get_post_analyze_result`
- MCP:
  - `gui_adapter_post_analyze_spec`
  - `gui_tab_get_post_analyze_params`
  - `gui_post_analyze`
  - `gui_post_analyze_wait`
  - `gui_post_analyze_poll`
  - `gui_tab_get_post_analyze_result`
- 若 post figure 要可取圖，決定沿用 `tab.get_current_figure` 還是新增 post figure 專用 screenshot/export surface。

新增 wire method 時需 bump `WIRE_VERSION`；純 MCP convenience 需 bump `MCP_VERSION`。

### Medium: PostAnalyzeService 缺 adapter capability gate

**位置**

- `lib/zcu_tools/gui/app/main/services/post_analyze.py:71`
- `lib/zcu_tools/experiment/v2_gui/adapters/base.py:194`

**問題**

`PostAnalyzeService.start_post_analyze()` 只 gate：

- tab not busy
- primary `analyze_result` exists

它沒有檢查 `tab.adapter.capabilities.post_analysis`。因此非 opt-in adapter 若被呼叫，會開 worker，最後由 `BaseAdapter.post_analyze()` 的 `NotImplementedError` 失敗。

**影響**

- 錯誤發生在 worker，回報較晚。
- operation handle 已建立，UI 進入 analyzing 再失敗。
- 違反註解宣稱的「adapter sets True AND primary result exists」。

**建議**

在 `start_post_analyze()` 開 handle 前 fail-fast：

```python
if not tab.adapter.capabilities.post_analysis:
    raise RuntimeError(f"Adapter {tab.adapter_name!r} does not support post-analysis")
```

並新增測試：非 post-analysis adapter 即使有 primary analyze result，也不應 submit worker。

### Medium: Terminal slot 的 service-side exception 可能留下 busy / live operation

**位置**

- `lib/zcu_tools/gui/app/main/services/analyze.py:144`
- `lib/zcu_tools/gui/app/main/services/post_analyze.py:125`

**問題**

worker 成功後，terminal slot 還會做 service-side work：

- `AnalyzeService._on_analyze_finished()`:
  - teardown old writeback editors
  - compute writeback items
  - update State
  - clear analyzing
  - settle handle
- `PostAnalyzeService._on_post_analyze_finished()`:
  - read `post_result.figure`
  - update State
  - clear analyzing
  - settle handle

若這些 terminal slot 內部丟例外，`is_analyzing` 可能不會清掉，handle 也不會 settle。這類錯誤不會走 worker `on_error` path。

**影響**

- tab 永遠 busy。
- `active_operation_count()` 永遠非零。
- shutdown / wait / poll 行為錯亂。

**建議**

terminal slot 應包住 service-side post-processing，確保失敗時：

- `set_tab_analyzing(tab_id, False)`
- `_release(OperationOutcome("failed", str(exc)))`
- emit `TabInteractionChangedPayload`
- emit `analyze_failed` / `post_analyze_failed`

同時保留原始 exception log。

### Low: Post-analysis result 的 user-facing surface 不完整

**位置**

- `lib/zcu_tools/gui/app/main/ui/main_window.py:1138`
- `lib/zcu_tools/experiment/v2_gui/adapters/singleshot/ge.py:130`

**問題**

post-analysis result 存在 State 中，但 UI 目前主要只展示 post figure，沒有 scalar summary、post result save/writeback flow，也沒有 remote readback。

`GEAdapter.guide()` 說 complex centers deferred to post-analysis phase，但目前 post-analysis result 仍只透過 `to_summary_dict()` 跳過 complex 欄位，也沒有 writeback path 寫 `g_center` / `e_center`。

**影響**

- 使用者或 agent 可能以為 post-analysis 能處理中心點寫回，但實作沒有。
- post-analysis 的價值目前主要是可視化，不是完整可操作結果。

**建議**

先釐清產品語意：

- post-analysis 是否只是一個可視化 second fit？
- 是否要提供 scalar result summary？
- `g_center` / `e_center` 是否要用 GUI 支援 complex MetaDict writeback，或改成拆成 real/imag float？

若暫不支援，請修正 guide 文字，避免承諾不存在的流程。

## 已驗證項目

### 通過

```bash
uv run pytest tests/gui/services/test_post_analyze.py tests/gui/services/test_analyze.py tests/gui/ui/test_post_analyze_ui.py tests/experiment/v2_gui/adapters/singleshot/test_ge_adapter.py tests/experiment/v2_gui/adapters/test_post_analysis_default.py
```

結果：42 passed。

```bash
uv run pyright
```

結果：0 errors, 0 warnings, 0 informations。

### 未能用 `.venv/bin/python` 跑 pytest

```bash
.venv/bin/python -m pytest ...
```

失敗原因：`.venv` 中沒有安裝 `pytest`。

## 建議修正順序

1. 先修 analyze/post-analyze token lifecycle：決定全域互斥或 per-tab token map，補並發測試。
2. 補 post-analysis capability gate，避免非 opt-in adapter 進 worker。
3. 強化 terminal slot exception handling，避免 busy / handle leak。
4. 補 remote/MCP post-analysis surface，讓 agent 與 Qt UI 能走同一功能。
5. 修正或補完整 post-analysis result summary / writeback / guide 語意。

