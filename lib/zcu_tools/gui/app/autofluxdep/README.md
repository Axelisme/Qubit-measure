**Last updated:** 2026-06-12（autofluxdep real-acquire migration / RB-0..RB-4）

# gui/app/autofluxdep/ — autofluxdep-gui app shell

控制型 GUI：拿某 flux 的 base context 衍生、自動跑 flux × node-graph 掃描。**domain core**（Builder/Node/orchestrator/Result/Patch/Plotter 的設計語言）見 `CONTEXT.md` + ADR-0018；本 note 記 **app shell**（session 服務組合 + dialog + run path）。

## 結構

```
autofluxdep/
├── controller.py    — Controller façade：組 build_session_services + 實作 SessionControllerPort（委派 conn/ctx/dev/startup/progress/predictor）；workflow 編輯命令；start_run/stop_run/dry_run；_build_tools（exp_context.predictor→FluxoniumPredictorAdapter / None→SimplePredictor）；_MlModuleSource（ModuleLibrary→ModuleSource proxy，None-on-absent + forward make_cfg/get_waveform）
├── state.py         — AutoFluxDepState(SessionState)：nodes/flux_values/flux_device_name/run_results/run_predictor + ProjectInfo；version keys workflow/flux
├── app.py           — build_core(project?, project_root?) + run_app（注入 repo-root project_root，組 MainWindow）
├── event_bus.py     — EventBus=BaseEventBus + autofluxdep payload（Workflow/Flux/Run* — SetupDone 已退）
├── operation_gate.py— app-local OperationGate（str-keyed 衝突矩陣 over session kinds + 自己 RUN kind）
├── background.py    — 瘦 BackgroundService（組合共用 BackgroundRunner；_entered 只 pbar+ActiveTask，無 figure routing，ADR-0018）
├── ui/              — main_window / node_list（Setup/Devices/Predictor button + flux-source picker）/ node_detail / param_form
├── nodes/           — **domain core（別動）**：builder（Builder/Node/RunEnv）/ qubit_freq/lenrabi/ro_optimize/t1/t2ramsey/t2echo/mist/predictor / acquire（共用 real-acquire helpers：set_flux_by_name/require_flux_device/build_stop_checkers/is_good_fit/parse_linear_axis）/ result / plotters / io / spec
├── orchestrator.py / derivation.py / tools.py / registry.py — **domain core（別動）**
└── CONTEXT.md / README.md
```

## 關鍵設計

- **複用 session core**：Controller `__init__` 組 `build_session_services`，注入 app-local `OperationGate` + 瘦 `BackgroundService` + 共用 `OperationHandles`/`ProgressService`(default `QtProgressTransport`)/`IOManager`。實作 `SessionControllerPort`（pyright 在 dialog call site 驗 conformance），開共用 `gui/session/ui` 的 setup/device/predictor dialog。run 讀 `exp_context`（soc/soccfg/ml/md/predictor），無自己的 SetupResources/setup()（早退）。
- **State main-thread 不變式 + Qt 需求**：`build_core()` 建 session-service QObject，**需 QApplication 先存在**（測試 conftest `qapp` autouse）；headless 測試經 `connect_mock`（async ConnectionService + QEventLoop）建 mock soc，不用真 setup。
- **Run path（real acquire）**：每 node `produce` 用 `Builder.make_cfg`（決策 A，D1：在 produce 跑）lower context→真 cfg，把這點 flux 值寫進 `cfg.dev[flux_device]`（`set_flux_by_name`，by device **name**＝picker 存的，非下層 `flux_dev` label）→ `setup_devices` 推送 → 建 program（TwoToneProgram/ModularProgramV2…）`.acquire`（`round_hook`+`stop_checkers`：cooperative stop＋SNR early-stop）→ fit（qubit_freq 並 `predictor.calibrate`；mist/ro_optimize 無 fit）。**無 synthetic fallback**：context 未配置時 `make_cfg` **Fast Fail**（→ orchestrator 轉 `RunFailedPayload`，不讓 worker QThread abort）。offline 測試走 flux-aware MockSoc（`connect_mock` 起 `fake_flux`），資料隨 flux 變化。詳見 `CONTEXT.md` 的 "Run path"。
- **produce 例外保護**：`orchestrator.run` catch `produce` 例外 → 記 `self.run_error`＋停 sweep；`controller.start_run` 檢查後 emit `RunFailedPayload(message)`，UI `MainWindow._on_run_failed` 解鎖＋彈 warning。真 GUI run 的 node 例外不會 crash app。
- **predictor 兩層**：`exp_context.predictor` 存 raw `FluxoniumPredictor`（PredictorDialog/ConnectionService 載）；`_build_tools` 每 run wrap 成自適應 `FluxoniumPredictorAdapter`（None→`SimplePredictor`），stash 進 run-lived `state.run_predictor`。**勿混** measure 的 freq-預測 predictor。

跨模組設計見 ADR-0017（worker 畫圖 marshal，本 app 不適用——worker 不畫）/0018（orchestrator 需求解析器）/0020（session-core extraction）。
