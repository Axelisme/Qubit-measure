# autofluxdep Phase B — 真實量測接入計劃

**Status:** 規劃中（投查完成，待用戶定中心架構決策再實作）。**Branch:** gui2（接續 session-core extraction `session_core_extraction.md`，S1–S5 reshape 已達標）。
**前提：** 動 domain core（`nodes/`、`orchestrator.py`）需用戶 sign-off；改 `lib/zcu_tools/gui/` 外（`experiment/v2/*`）需批准（read-only import 不需）。

## 目標

把 autofluxdep flux-sweep 從 **synthetic 訊號**（Phase C 現況）接到 **真實硬體量測**：每個 flux 點先設 flux device 值，每個 node `produce` 真的對 SoC acquire（取代合成）。

## 關鍵投查結論（2026-06-10，兩 opus agent 精確勘查）

### 🔑 最重要：下層已有真實實作（非從零）
`experiment/v2/autofluxdep/`（**非 GUI**）已是**可跑的真硬體 outer-loop flux sweep**：
- `executor.py:242-309 FluxDepExecutor.run` — outer Python loop `.scan("flux", flux_values, before_each=update_fn)`，`update_fn` 末呼 `set_flux_in_dev_cfg(ctx.cfg.dev, flux, label="flux_dev")`；每點先 prime `flux_values[0]` + `setup_devices(progress=True)`。
- `qubit_freq.py:93-121` per-node `measure_fn` = `setup_devices(cfg)` + `TwoToneProgram(soccfg, cfg, sweep=[("detune", sweep)]).acquire(soc, round_hook=, stop_checkers=[ctx.is_stop, ...])`。
- measure-side 雙生：`twotone/fluxdep.py FreqFluxExp`（`FluxDepAdapter.exp_cls`）。

GUI `gui/app/autofluxdep/`（orchestrator + nodes）是**平行 reimplementation**：有更豐富的 dependency model（Snapshot/Patch/derivation/Tools 跨點平滑）+ GUI liveplot，但 `produce` 目前是合成（`nodes/synth.py` 的 `lorentzian_dip`+`accumulate_rounds`）。

### per-point flux 是 outer Python loop（非 program sweep）— 已證實
program 內部（FPGA）只 hard-sweep 內軸（freq/detune）；flux 由外層 Python loop 每點 `set_flux_in_dev_cfg`+`setup_devices` 驅動。GUI orchestrator 的 flux loop 正是這個外層。**計劃原定方向正確**。

### cfg.dev 結構 + flux 寫值機制
- `cfg.dev: Mapping[str, DeviceInfo]`，key=device name，`DeviceInfo.label="flux_dev"`。
- `set_flux_in_dev_cfg(devs_cfg, value, label="flux_dev")`（`experiment/utils/device.py:29-41`）按 `.label` 找 device 改 `.value`；`setup_devices(cfg)`（`device.py:91-97`）→ `GlobalDeviceManager.setup_devices` 推到硬體。
- cfg 由 adapter lowering 組：`CfgSchema.to_raw_dict(md, ml)` → `build_exp_cfg` → `ml.make_cfg(raw, ExpCfg)`（`meta_tool/library.py:136-167`，融合 md/ml/dev）。`FluxDepAdapter`（`adapters/twotone/flux_dep.py`）是最接近的現成 flux-sweep adapter。

### 測試性：MockSoc 可跑真 acquire
measure 全功能 MCP smoke（mock connect）已跑真 twotone acquire 全綠 → **Phase B real-acquire node 可 headless 測**（mock connect + MockSoc，免真硬體）。

## GUI domain core 的 5 個整合 gap（精確 seam）

| # | seam | 現況 | 落點 |
|---|------|------|------|
| 1 | per-point flux-set | **缺** | ~~orchestrator loop hook~~ **改：flux 值由 per-point `build_node` baked 進 `cfg.dev`，`produce` 的 `setup_devices(cfg)` 推送**。orchestrator loop 不動 |
| 2 | synth → real acquire | synth `qubit_freq.py:159-167` | 換成 `TwoToneProgram(soccfg, cfg, sweep).acquire(soc, round_hook=env.round_hook, stop_checkers=...)`；round_hook 已對映 |
| 3 | cfg builder（cfg_maker）| **GUI 完全無** | 需引入：node `produce` 內建 cfg（從 ml/md/flux）— 複用 measure adapter 機制 vs 自寫（決策 B）|
| 4 | `ml=` 進 Orchestrator | start_run **沒傳**（`controller.py:524-530`），dry_run 有 | start_run 補 `ml=self._state.exp_context.ml`；若 node 要 `ml.make_cfg`，ml 還要進 **RunEnv**（新欄位，目前只到 `project_snapshot`）|
| 5 | soccfg / flux-device handle / dev-cfg | **未 thread**：RunEnv 有 `soc` 無 `soccfg`/`ml`/`dev` | RunEnv/Orchestrator 加 soccfg + dev-cfg（或 flux device name）；soccfg 在 `exp_context.soccfg`（start_run 沒讀）|

（`soc` 已在 `RunEnv.soc`，`orchestrator._make_env` 已 curry。）

## 架構決策（用戶定 2026-06-10）

**決策 A = port-into-nodes，cfg 在 Builder.build_node 建好注入 Node**：
> NodeBuilder 把「設定頭」（PlacedNode params）透過 **per-Builder cfg_maker** 轉成 cfg，在 `build_node` 初始化 Node 時傳入；Node `produce` 用該 cfg 真 acquire。保留 GUI dependency model（Snapshot/Patch/derivation/predictor 閉環）。下層 `FluxDepExecutor` 退役與否暫不決（先讓 GUI 路徑成形）。

**決策 B = 先 B2**：autofluxdep 自寫輕量 cfg_maker（直接組 `TwoToneCfg` 家族 cfg），不拖 measure 的 CfgSchema/adapter 機制。**per-Builder**（每個實驗 node 的 cfg/program 不同 → 各 Builder 擁有自己的 cfg_maker + program class，仿下層 `QubitFreqTask` 持 `cfg_maker` + `TwoToneProgram`）。

**🔑 設計細化（讀下層 `experiment/v2/autofluxdep/qubit_freq.py` QubitFreqTask.run 得）**：cfg **依賴 `predict_freq`**（predictor 預測，設成 `qub_pulse.freq` 中心 + detune sweep recenter，見 `qubit_freq.py:137-165`）。`predict_freq` 在 GUI 是 **snapshot** 內容（predictor Service 產，只在 `produce` 拿得到，build_node 沒有）。故 A 細化為：
- `build_node(env)`：`cfg = self._make_cfg(env.params, env.ml, env.flux)` 建 **base cfg**（modules from ml、reps/rounds/sweep-range from params、`cfg.dev` 把 **這點 flux 值** baked 進 flux device，因 build_node 是 per-flux-point）→ `Node(env, cfg)`。
- `produce(snapshot)`：把 `snapshot["predict_freq"]` patch 進 `cfg.modules.qub_pulse.freq`（中心）+ recenter detune sweep → `setup_devices(cfg)`（推 flux 到硬體）→ `TwoToneProgram(env.soccfg, cfg, sweep).acquire(env.soc, round_hook=env.round_hook, stop_checkers=[...])` → fit → `env.tools.predictor.calibrate(flux, fit_freq)` → Patch。

**✅ 簡化：orchestrator RUN LOOP 不動**。因 flux 值由 per-point `build_node` baked 進 `cfg.dev` + `produce` 的 `setup_devices` 推送，**不需在 orchestrator loop 插 per-point flux hook**（原 gap #1 消解）。domain-core 改動縮到：`RunEnv`(builder.py 加 soccfg/ml 欄位) + 各 node `build_node`/`produce` + `orchestrator._make_env`(thread soccfg/ml) + `controller.start_run`(傳 soccfg/ml)。

**待確認（C）**：`ModuleLibrary` 是否滿足 orchestrator 的 `ModuleSource`(`get_module`)；cfg_maker 是否還需要 `md`（暫假設 ml 的 concrete modules + params + flux 夠，需要再加 RunEnv.md）。

## ⚠️ 重排（用戶 2026-06-10：「先一律 synthetic，真 acquire 最後實作」）

produce **暫不換** real acquire（real acquire vs MockSoc 只回噪音，synthetic 路徑正是為此存在、~140 測試/demo 全靠它）。**真 acquire stack 全延到最後**：B-3（其他 node cfg_makers，只被 real acquire 用）+ B-5（flux picker，要驅動 flux 才有意義）+ produce real-acquire swap（mock/real 分支）。**先做不依賴 acquire 的**：B-1✅（cfg_maker 已 ready 但暫不接 produce）→ **B-4 predictor 載入 UI**（synthetic run 也讀 exp_context.predictor）→ B-6 cleanup。real acquire 切換待用戶定 R1（mock→synthetic / real→acquire，偵測切換）vs R2，傾向 R1+自動偵測。

## Phased 分解（決策 A + B2 定後）

> 每 phase 獨立綠 + 可 commit；real-acquire node 用 mock connect + MockSoc 測（免真硬體）。動 domain core 每 phase 具體 diff 先給用戶過目。

- **B-0 接線基礎**（additive 低風險）：`RunEnv` 加 `soccfg`/`ml` 欄位（builder.py）；`orchestrator._make_env` thread `soccfg=self.soccfg, ml=self.ml`（Orchestrator 加 `soccfg` 欄位，`ml` 已有）；`controller.start_run` 讀 `exp_context.soccfg`/`.ml` 傳進 Orchestrator。**produce 仍 synth，只把料備齊**，全測試不變。*動 builder.py/orchestrator.py/controller.py — domain core。*
- **B-1 per-Builder cfg_maker（B2）**：每個 Builder 加 `_make_cfg(params, ml, flux) -> base cfg`（純函數，組 `TwoToneCfg` 家族：modules from ml、reps/rounds/sweep-range from params、`cfg.dev` baked 這點 flux）。先**純函數 + 單測**（不接 acquire）。先做 qubit_freq 一個。
- **B-2 vertical slice（qubit_freq 端到端真 acquire）**：`build_node` 呼 `_make_cfg` 注入 Node；`produce` patch `snapshot["predict_freq"]`→`qub_pulse.freq` + recenter sweep → `setup_devices(cfg)` → `TwoToneProgram(soccfg, cfg, sweep).acquire(soc, round_hook, stop_checkers)` → fit → `predictor.calibrate`。mock connect + MockSoc 跑通一條。**真正風險點，先打通一個 node**。stop_checkers 接 GUI 的 stop（`ctx.is_stop` 等價）。
- **B-3 roll out**：lenrabi/ro_optimize/t1/t2ramsey/t2echo/mist 各 node 比照（每 node 一個 commit + cfg_maker + produce real）。
- **B-4 predictor 真載入 UI**（Fork 1 暫緩的）：promote/share measure `PredictorDialog`（`gui/app/main/ui/predictor_dialog.py`）成共用 + autofluxdep Controller 曝 `load_predictor`/`predict_freq`；node_list 加 Predictor button。`exp_context.predictor` 存 raw FluxoniumPredictor，`_build_tools` 已會 wrap。
- **B-5 flux-device picker UI**：選哪個 connected device 當 flux 源（→ cfg_maker baked dev 的 device name/label），退「寫死 flux_dev label」假設。
- **B-6 收尾**：app.py 注入 `project_root`（目前 default cwd 可用）；文件/AI_NOTE 更新；決定下層 `FluxDepExecutor` 去留。

## Scope / 批准矩陣

| 動的東西 | 類別 | 需批准？ |
|---|---|---|
| `orchestrator.py`、`nodes/*.py`、`nodes/builder.py`(RunEnv) | autofluxdep domain core | **是（handoff 禁區）** |
| `controller.py`、`ui/*`、predictor/flux-picker UI | gui session/UI side | 否（gui scope 內）|
| `TwoToneProgram`/`setup_devices`/`set_flux_in_dev_cfg`/`ml.make_cfg` | 下層 experiment/v2 | read-only import OK；若需**改** → **是** |
| `experiment/v2/autofluxdep/FluxDepExecutor` 退役/改 | 下層 | **是（非 gui edit）** |

## 風險 / 開放問題

1. **決策 A（port vs delegate）+ 下層退役** — 最大架構叉，決定整個 Phase B 形狀。
2. **GUI node 與下層 task 的 dependency-model 對齊** — GUI 的 Snapshot/Patch/derivation 是否能餵真 acquire 的輸出，且跨點 predictor 校準閉環不被破壞。
3. **cfg_maker 複用 measure adapter vs 自寫**（決策 B）— 影響 autofluxdep 要不要拖整套 CfgSchema/spec 機制。
4. **MockSoc acquire 的回傳** 能否被現有 node fit 路徑吃（synth 與 real 的 data shape 對齊）。
5. **stop/cancel**：下層用 `ActiveTask` 全域 stop flag + `ctx.is_stop` in `stop_checkers`；GUI BackgroundService 已進 ActiveTask scope（worker stop_event）。real-acquire node 要把 `stop_checkers` 接上 GUI 的 stop。

## 建議起手

決策 A/B/C 定後，從 **B-0（接線，gui-only 低風險）** + **B-3（一個 node vertical slice）** 打通一條真路徑驗證可行，再 B-4 roll out。動 domain core 前每個 phase 的具體 diff 先給用戶過目。
