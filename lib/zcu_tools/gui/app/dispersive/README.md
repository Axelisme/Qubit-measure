**Last updated:** 2026-06-25（Fluxonium Prediction engine delegation）

# `zcu_tools/gui/app/dispersive/` — Fluxonium Dispersive-Shift Analysis GUI AI Note

> **MCP 搬遷（2026-06-08, c8eb1a03）**：MCP server entry 與共用傳輸 `McpBridge` 已搬出 `gui/` 到 `zcu_tools/mcp/`——`McpBridge`→`zcu_tools/mcp/core/bridge`、本 app entry（原 `services/remote/mcp_server.py`）→`zcu_tools/mcp/dispersive/server.py`；`MCPBridgeConfig` 拆出基底 `McpServerConfig`（無 launch 欄位）。`mcp` 是 `gui.remote` wire 層的**使用方**（非 leaf）。本筆記內 `gui/remote/mcp_bridge`、`(services/remote/)mcp_server.py`、`parents[6/7]=lib/repo`（現為 `parents[3/4]`）等舊位置/深度按此對映。

> **位置**：第三個 tool_gui app，與 `app/fluxdep/`、`app/main/` 並列。import 一律 `zcu_tools.gui.app.dispersive.X`；同子目錄內相對、跨子目錄絕對。「以後不再搬遷」。

## Module Purpose

獨立的分析型 GUI，把 `notebook_md/analysis/dispersive.md` 的 fluxonium **色散位移（dispersive shift）** g / bare_rf 擬合流程移植成 Qt 桌面工具。是 fluxdep-gui 的 sibling：自己的 state / services / UI / RPC + MCP server / skill。

**領域依賴 fluxdep**：兩 app 經 **同一個 `params.json` 的不同 section** 銜接 —— fluxdep-gui 寫 `fluxdep_fit`（EJ/EC/EL + flux 對齊），dispersive 讀它當輸入、寫 `dispersive`（g, bare_rf）。典型工作流：先跑 fluxdep-gui，再跑 dispersive-fit-gui。`update_result` 保留 fluxdep_fit（不用 dump_result，會覆蓋）。

**簡化過（移除 chi 圖 + qub_dim/cutoff 控件 + step）**：流程是 **load inputs → load onetone → preprocess → tune g/r_f（手動 slider 或 Auto tune）→ export**。**fit 由手動 accept 定案**：user 調 g/r_f（拖 slider，或先按 Auto tune 讓 scipy 粗調，見下方 step4 段）後按「Use these g/r_f」即最終 fit（`set_manual_fit` 記 State）；Auto tune 只回填 slider、不自動 accept。**沒有 chi 圖 / Result tab**：tune 圖（g/r_f 線疊 norm-phase）即最終結果。`qub_dim`/`qub_cutoff` 寫死在 `PredictService` 的 `PredictionResolution`（`qub_dim=15`/`qub_cutoff=30`/`res_dim=4`），不曝露為控件。**沒有 step**：預測一律跑全 preprocessed flux 軸（preprocess 已降採樣、fast 路徑夠快），`DispFitState` 無 step、`fit.result` RPC 無 step。

## 架構（對標 fluxdep，骨架共用、領域各寫）

分層：`app.py`（composition root，bootstrap 走共用 `run_qt_app`/`gui/run_app`，`run_app` 塌成 factory 接線）→ `state.py`（被動 DispersiveState + VersionTable）→ `controller.py`（命令 façade，emit EventBus；**繼承共用 `BaseController`**/`gui/controller_base` 取 state/bus/project_root + `_emit`，façade body 各 app）→ `services/`（純 sync Qt-free）→ `services/remote/`（read-only RPC + MCP）→ `ui/`（**單流程面板**）→ `event_bus.py`。

**與 fluxdep 的關鍵差異**：dispersive 是**單 onetone、單一線性流程**，不是 fluxdep 的「多 spectrum 集合 + 左 list 右 stacked」。`PipelinePanelWidget` 佈局：**控件壓在上方緊湊排**（step1/2/3 並排一排，**只放按鈕**）、**底下一條共用 info bar**（`_build_info_row`，把 step1 inputs 文字 + step2 onetone 狀態移來這，讓 step1/2/3 box 變矮）、再 step4 tune、**最下一個 QTabWidget 共用大圖區**（Preprocess / Tune 兩 tab）。前一步完成才 enable 下一步。step3 preprocess / step4 tune 各自獨立 **busy/indeterminate** progress bar（compute 都是黑盒，無百分比），`_begin_progress`/`_end_progress` 顯/隱、`_active_progress` 記當前。**step4 自身佈局**：上方 g / r_f 兩條 QSlider **垂直排**（各一列：名稱+slider+數值）；下方一橫列左 = Add sample flux / Clear samples / **Auto tune** / Use these g/r_f，右（push）= Export 按鈕（+label）。**沒有獨立 step5**：export 收進 step4 右下角（`_export_btn` gate on has_result，非整個 box）。`_inputs_label`/`_onetone_label` 在 info bar（不在 step1/2 box 內）。

**load onetone 有 live preview**：`ui/load_dialog.py`（`LoadOnetoneDialog`，**subclass 共用 `gui/widgets/load_dialog.LoadDataDialog`** —— 共用 file row/transpose/preview/OK-gate，`_build_options` 不加額外 widget、`result_request` 回 2-field `LoadOnetoneRequest`；去掉 fluxdep 的 type/inherit）。best-effort `load_data` 讀檔 + imshow 預覽，transpose toggle 翻 preview x/y，user 看著判斷軸要不要轉。transpose flag 只在 Load 傳給 Controller（State 永遠存 canonical x=flux/y=freq）。

### 共用層（與 fluxdep / measure 三方共用）
- `zcu_tools.gui.version_table.VersionTable`（直接依賴：`state.py`）。
- `zcu_tools.gui.remote.{errors,framing,param_spec,wire}`（傳遞依賴——經共用 transport 層帶入，dispersive 不直接 import）。**dispersive 的 worker 不畫圖（compute/record 模式），故不依賴 `zcu_tools.gui.plotting`（ADR-0017 R4 不適用）。**
- `zcu_tools.gui.remote`：transport 機制三方共用。RPC 方法登錄 `MethodSpec/BoundMethod/build_method_registry`（`method_spec.py`）；GUI 側傳輸 `NdjsonRpcEndpoint`（`rpc_endpoint.py` —— socket 生命週期 + accept loop + NDJSON framing + per-client writer/outbound queue + 內建 `wire.version`/auth handshake + reply 編碼 + `broadcast` push fan-out + 主執行緒 marshal primitive + `ClientLink`）；MCP-server 側傳輸 `McpBridge`（住 `zcu_tools.mcp.core.bridge` —— socket 狀態為 instance attrs、send_rpc_raw/connect/disconnect/launch/stop/reader thread/RID routing + 注入 on_event hook）。
- `zcu_tools.gui.event_bus`：`BaseEventBus`/`BasePayload` 三方共用機制。
- wire 版本常數 `services/remote/wire_version.py` 留各 app（dispersive **WIRE=3/GUI=3**：v3 fit.result 去 step）。每 app 各自演化 wire 契約故不入共用 `remote.wire`。

### 領域各 app 自有（共用機制之上）
- `event_bus.py`：dispersive 的 payload 型別 + `EventBus`（建在共用 `BaseEventBus` 上）。
- `services/remote/service.py`（RemoteControlAdapter）：**subclass 共用 `RemoteControlServiceBase`**（`gui/remote/control_service`），router scaffolding（route / events.* / `_dispatch_on_main` bare marshal / EventBus subscribe/serialize/broadcast，底層委 `NdjsonRpcEndpoint`）全在 base。dispersive read-only → **零 policy 覆寫**（`_get_bus` 用 base 預設 `ctrl.bus`、serializers 以 payload `type` 為 key），本檔只剩 domain 注入（method registry / serializers / 版本 / `server_name="DispersiveRemoteServer"`）。
- `ui/error_messages.py`：用共用 `gui/error_messages` framework（`normalize_raw`/`details_tail`/`friendly_from_rules`/`fit_io_redirect`），domain rule（`friendly_io_message` + `_FIT_RULES`）各 app。其餘領域新寫。**`ProjectDialog` 已抽共用**（`gui/widgets/project_dialog.py`，`db_label="One-tone dir"`）；`ProjectInfo`/`default_*`/`nearest_existing` 在共用 `gui/project.py`（Qt-free）。本目錄 `ui/project_dialog.py` 已刪。

## 計算全走 worker（避免 GUI 卡頓）
**重計算都經共用 `gui/background.py` `BackgroundRunner.submit`（per-panel，`enter=None`——無 routing/pbar scope）** off-main：preprocess（joblib edelay）、predict（tune 圖，scqubits）、auto_tune。模式統一：提交的 `work` 純呼 `compute_*`/`predict_*`（讀 State 不寫）並回**純資料 dataclass**（`_TuneData`/PreprocessResult）；`on_done`（主執行緒）唯一 `record_*`（State 寫）+ **畫圖**（worker 從不碰 Qt widget / pyplot，守 ADR-0017 + main-thread State 不變式）。一次只跑一個用按鈕 disable guard（不需 generation 戳記）。

- **step4 Tune 互動模型**：**g=QSlider**（固定 0..200 MHz、1 MHz/tick、預設 50，`_G_MIN/MAX/DEFAULT_MHZ`，`_g_mhz()` 讀值）、**r_f=QSlider**（MHz 整數）；兩條 slider 垂直排（`_slider_row` 名稱+slider+數值）。**res_dim/qub_dim/qub_cutoff 全寫死**（`PredictionResolution(res_dim=4, qub_dim=15, qub_cutoff=30)`）。
  - **preprocess done → `_init_tune_view`**：r_f slider 用**固定 0..300 tick**（`_RF_TICKS=300`）跨數據 sp_freqs.min~max → 精度恆 = **span/300**（不隨範圍寬窄變）；tick↔GHz 映射在 `_rf_ghz`/`_rf_tick_for`。預設 tick = 最接近 `PreprocessResult.median_rf`（每 flux norm_phases 峰值頻率的**中位數**，在 preprocess 算）。tune 圖**先畫背景熱圖 + r_f 線**（`render_tune_figure`，無色散線）。同時 `bind_drag` 把 TuneCanvas 的 sample-line 拖曳指向新 artists。
  - **拖 r_f / g slider（debounce）**：line / 數值 label 每 tick 即時動（純畫、不算），但 sample 點重算 **debounce `_SLIDER_DEBOUNCE_MS=150ms`**（`_dot_debounce` single-shot QTimer，slider move 只 `start()`，停手才 `_on_dot_debounce_fired`→`_refresh_sample_dots`）。
  - **按 Use these g/r_f → `_PredictWorker`**（off-main fast/scqubits）→ done slot `set_dispersion_lines` 畫 ground/excited 色散線 + `set_manual_fit` 記結果（**手動 tune 即最終 fit**）+ enable export。**計算中 disable 按鈕**（一次只跑一個、不需 generation guard）。
  - **sample-flux 線（即時、不整體重算）**：「Add sample flux」在 flux 軸中央放一條**可滑鼠拖曳的垂直線**（`TuneCanvasWidget` 用 `mpl_connect` press/move/release，`_pick_sample` 用 x-tolerance 抓最近線）；每條線顯示該單一 flux 的 **ground(藍)/excited(紅)** 點（**與色散線同色**：dispersion line ground=`b-`/excited=`r-`，dots 對齊）。**拖線 motion 只移線不算**（`_on_sample_drag`），**release 才重算該點**（`_on_sample_drop`，`bind_drag(on_drag,on_drop)`、canvas `_dragged` 旗標只在真的有動過才觸發 drop）。`_compute_sample_dots`=**batched 單點 `predict_sample_points`**（繞 `PredictService` 軸綁定快取，走 engine stateless `predict_dispersive_at`），**不跑全軸**。fresh preprocess（`_init_tune_view`）清空 sample 線；「Clear samples」原地移除（保留色散線）。
  - **Auto tune（scipy，off-main，global→local）**：`services/autotune.py` 的 `auto_tune` 先 `_coarse_seed`（**2D r_f×g 粗網格** `_COARSE_N_RF=50`×`_COARSE_N_G=10` + 當前 g0/rf0 也當候選，取最高 `sample_score` 點）當種子，再 `scipy.optimize.minimize`（Nelder-Mead，手動 bound clamp）局部精修。**為何要粗掃**：純局部會卡在 decoy（偽亮帶）的 basin（實證：種子落 decoy 5.2 附近時純局部 rf=5205 WRONG，粗掃跳出找到真值）；乾淨雙帶 case 純局部本來就 OK，但真實譜有 spurious feature 故加全域。grid+NM ≈ differential_evolution 同準同速但**確定性、可控、透明、複用 predict**（評估過 brute/diff_evo 都不選）。loss = mean over sample fluxes of `max(norm_phase@ground, norm_phase@excited)`，**最大化**（`norm_phase` 用 `RegularGridInterpolator` bilinear、查詢點 **clip 到網格邊界免外插負值**）。`controller.auto_tune` 快照 State 純算 → `_AutoTuneWorker` off-main（粗掃 ~500 predict ≈ 2-3s）→ done slot **只回填 g/r_f slider**（不 accept、不畫全軸；user 自己再按 Use these）。**無 sample 線時按鈕 disable**（`_sync_auto_tune_enabled`，add/clear 時同步，因 sample 線在 artists 非 State）。⚠ **g 常被 loss 的 `max()` 弄成不可辨**（r_f 對後只要 ground 或 excited 之一落在亮帶，max≈1、g 幾乎不影響 score → g 易飄到 bound）；r_f 收斂良好。這是用戶自訂 loss 的固有性質、非 bug，且只回填供 user 檢視。
  - viz：`render_tune_figure`（背景+r_f 線）/ `update_bare_line`（slider 即時）/ `set_dispersion_lines`（predict 後色散線）+ sample 四件 `add_sample_line`/`move_sample_line`/`remove_sample_line`/`update_sample_dots`（`TuneArtists.samples: list[SampleArtists]`，各帶 dot_ground/excited Optional）。

> **R4 不適用**：worker 只回資料不在 worker 畫圖 → 不需 routing_scope。

## 效能關鍵
- `PredictService`：薄 adapter,固定 GUI `PredictionResolution` 後委派 `FluxoniumPredictionSession`。axis-bound cache、fast/scqubits fallback 與 backend provenance 屬 simulate engine；cache key 是 `(g,bare_rf,return_dim)`（**無 step**）。綁定一組 (params, flux-axis)，inputs/preprocess 變則 Controller 重建 service/session。
- **sample-flux 單點**走 `predict_dispersive_at`（不經 PredictService 軸綁定，arbitrary fluxs），同樣委派 engine 的 stateless dispersive prediction；GUI normal path 不 catch `DressedLabelingError`。
- **fast 函式的 flux-independent operators 由 simulate 端 `_fluxonium_operators` @lru_cache(params,cutoff,dim)**：scqubits `cos_phi_operator`/`sin_phi_operator` 走 scipy `cosm`/`sinm`(expm)，fresh Fluxonium 每次重建 ~84ms → cache 後 drag 單點 **84→0.6ms**（~130x），全軸 predict 也降到 ~11ms。numerically identical（只 memoize）。
- preprocess edelay 用 numba kernel（見「已知坑」段），predict 用 fast dispersive —— 兩個 scqubits/scipy 熱點都繞過了。

## 單位
**State / wire 全程 GHz**（g, bare_rf, bare_rf_seed, freqs）；slider UI 顯示 MHz（×1e3 進 / ×1e-3 出）。onetone load 經 `format_rawdata` Hz→GHz（**不走 experiment 層**：`FluxDepExp.load` 回 dataclass、MHz）。

## Remote / MCP（read-only，fluxdep 模式）
agent 只觀測、user 在 GUI 驅動。method set 全純查詢（state.check / project.info / fit_inputs.info / preprocess.status / fit.result{has_result,g,bare_rf,res_dim} / resources.versions），**無 mutating RPC**。`mcp/dispersive/server.py` 是 **thin entrypoint**：填 config + instructions，body（`send_gui_rpc` / 3 lifecycle tools launch/connect/disconnect（**無 stop 曝露**）/ cleanup / `run_stdio_loop`）全交 `mcp/core/readonly_server.build_readonly_server`；read-only tools 從 METHOD_SPECS 自動生成。**event-push 丟棄**（read-only 無診斷流，bridge on_event 不接）。預設 control port **8767**（避開 fluxdep 8766）。當 script 跑：絕對 import（無 parent package）+ lib path inject。**WIRE=3/GUI=3**（v3：fit.result 去 step）。`project.info` / `resources.versions` 直接註冊 `gui/remote/readonly_handlers.py` 的共用 `h_project_info` / `h_resources_versions`（與 fluxdep 共用，兩 app wire 形態永遠同步）；`_h_state_check` 仍 app-local（用 `gui/project.py` 的 `is_real_project`）。

## 已知坑 / 設計決策
- **preprocess edelay 用 numba kernel（GUI-local，非 utils）**：`services/_fast_edelay.py` 的 `@njit(parallel=True, fastmath=True, cache=True)` 把整個 (n_flux × 1000-grid) 雙層 loop 編成機器碼、per-flux 外層 `prange` 平行，內聯 **Kasa 代數圓擬合**（2×2 solve，取代 utils 的 `fit_circle_params` scipy.eig，圓心相同、半徑差 ~2e-3）。**~14x**（benchmark：scipy.eig+loky 1.49s → numba 8thr 0.103s；單核也 5x）。numba 放 GIL → 不 fork、不 pickle（解掉 `GuiProgressBar` 過 fork pickle 問題）；kernel 是黑盒 ~0.1s → 進度條走 **busy/indeterminate**（無 per-flux tick，`gui_pbar.py` 已刪）。**不動 utils 通用 `fit_edelay`/`fit_circle_params`**（這是 dispersive 專用快路徑）。數值對齊：transpose 正確的真實檔上 numba vs utils fit_edelay 差 = 0.000000。
  - **平行化教訓（benchmark 實測）**：per-row Python loop + threads **比 serial 慢 3.5x**（`np.linalg.solve`/scipy LAPACK 不放 GIL，線程互搶）；向量化（消內層 grid loop）後 threads/loky 才好（0.4-0.5s）；numba prange 最贏（放 GIL 的真平行 + 無 fork 開銷）。
- **OneTone 檔軸常反**：OneTone hdf5 常存成 `[Frequency, Flux]`，loader 假設 `[Flux, Flux→Freq]` → 載入後 freqs 變垃圾（如 ~1e-12），preprocess edelay 全錯。**不是 bug，是檔格式** —— user 在 load dialog 看 preview 勾「Transpose axes」修正（這正是 preview dialog 的用途）。同 fluxdep OneTone 軸反問題。
- **preprocess smoothing signature**：preprocess 在 edelay/circle-fit 前與 phase-diff 前沿頻率軸做 smoothing，預設 method 是 `wavelet`；signature 包含 method、兩個 smoothing divisor 與 grid shape，因此 smoothing pipeline 變更會讓既有 fit 失效。小頻網格的 smoothing strength floor 至 1，GUI 不因粗網格崩潰。
- **路徑**：`result_dir`=`result/<chip>/<qub>`（processed 輸出 / params.json）；`database_path`=`Database/<chip>/<qub>`（**raw onetone root**，對齊 notebook `Database/Q12_2D[5]/Q1/...`，**不是** result/ 下）。browse onetone 預設走 `database_path`。
- **export 需既存 params.json**：`update_result` 先讀檔；export 前 gate on `has_result` + 檔存在，缺則 fast-fail 指向 fluxdep。
- **bare_rf seed 不覆蓋**：`set_fit_inputs` 只在 disp_fit.bare_rf 無值時 seed（不蓋既有 tuning 值）。
