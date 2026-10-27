# `zcu_tools.gui.app.fluxdep` flux-dependence analysis GUI

**Last updated:** 2026-10-27

**讀者：** framework-dev。通量頻譜分析 GUI 的狀態、互動與搜尋分工。

MCP 組合根位於 `mcp_servers/fluxdep.py`，stdio 實作位於 `zcu_tools.mcp.fluxdep.server`；本 package 只包含 GUI app、
state/services/UI 與和 `ui/` 平級的 GUI-process remote driving adapter。Import path 固定為
`zcu_tools.gui.app.fluxdep.*`。

## Module purpose

獨立的分析型 GUI，承接 `notebook_md/analysis/fluxdep_fit.md` 的 fluxonium 能譜擬合
流程。**領域層獨立於 measure-gui**：自己的 state / services /
互動 widget / skill，與 measure-gui 領域零耦合。**共用機制（transport + event）位於
`gui/remote`（`NdjsonRpcEndpoint`、`MethodSpec`）與
`gui/event_bus`（`BaseEventBus`/`BasePayload`），三個 app 共用一份**；domain 仍各
app 自帶（version table、worker 模式等仍 per-app）。

這是獨立於 measure-gui 的可選分析工具。

## Pipeline

分析以決策點組織，步驟可反覆執行，順序由人或 agent 決定。資料依賴仍是對齊 → 採樣 → Filter → fit。下圖列出資料依賴，使用者可反覆操作各決策點：

```
載入 hdf5 spectrum(OneTone/TwoTone)
  → 定線(LinePicker)       → flux_half / flux_int / flux_period(per-spectrum, 可繼承)
  → 選點(OneTone slider / TwoTone brush mask)  → 每張譜的標註點
  → [累積多張 spectrum 進集合]
  → 跨譜篩選(Selector)     → 每張譜的 Filter 遮罩 + min_distance
  → 匯出 spectrums.hdf5
  → 搜資料庫(FitPanel) → (EJ,EC,EL) + 視覺化 + 診斷圖 → 匯出 component predictor 與 setup
```

與 measure 的根本差異（為何領域全新寫、不套 adapter/tab/run）：fluxdep 是
**單一 pipeline session**（spectrum 集合），不是平行多 tab；步驟間直接傳資料、
可反覆重做；不碰硬體（無 soc/device/reps/sweep）。

**只做 search，不做 scipy fit**（fit_spectrum 留在 notebook，未移植）。

## Entry and export

Project… 使用共用 EntryDialog 開啟 `project_root/result/<entry>`。CLI 使用 `--project-root`、`--entry`、
`--component`。未指定 component 時，app 解析 setup 的 `qubit` alias。
目標必須是 `qubit/fluxonium` 或其子 kind。共用 owner 回報不存在的 entry/component 或無效的 alias。
CLI 未指定 entry 時，GUI 保持未開啟。
Dialog 列出全部合法 component，有效 qubit alias 只作預選。
Alias 缺失或錯誤時，OK 保持停用，直到使用者明示選擇。
不建立新條目，不讀舊兩級目錄或 params.json。啟動腳本註冊 lab component schema，框架不 import lab。

Export params… 寫 `predictors/<component>.json`，再更新 setup 宣告的 EJ／EC／EL
與注入 planner 提出的校準欄位，最後追加 analysis ledger。
`export_plan.ExportPlanner` 接收 detached component 與 app owner 的 `AlignmentPublication`，包含 aligned native integer／half／period，
回傳 component-relative scalar writes 及 `flux_writeback`。未接 planner 時，export 在輸出前拒絕。
Framework 不解讀 lab 的 flux 區塊，也不取得 device 值。

`apps.fluxdep` 注入 [`zcu_lab.fluxdep_export`](../../../../../zcu_lab/fluxdep_export.py) 的規則。
有 `flux_dev` 時只更新 `flux.period.<flux_dev>` 的大小，保留原方向，首次使用正方向。
單儀器、多儀器、空 period mapping 與 offset=null 都不改 component offset，其他 periods 也保留。
Operator 必須確認 `flux_dev` 是這批資料的被掃儀器，app 不從 snapshot 推測來源。
缺 `flux_dev` 時跳過 period，但 EJ／EC／EL、predictor 與 ledger 仍匯出。
GUI status 與 RPC／MCP 的 `flux_writeback` receipt 維持 written／skipped、目標 device 與原因，不新增 partial。
資料可能經歷 flux jump，app 對齊不代表目前 component offset。
只有量測 fluxdep writeback 會另提預設勾選的 offset，扣除其他儀器在 Run snapshot 的貢獻。
無效名稱或擬合點在 planner 拒絕，不猜 device 或補值。

Predictor 的 `zcu.predictor` v1 格式與 `flux_half／flux_period／flux_bias` 模型座標不變。
Spectrum 的 reference anchors 與 processed checkpoint 也不是 component flux frame。
沒有來源 run 時 source.run_id 是 null。校正取第一張已對齊的 spectrum，包含零點來源。
這些檔案依序寫入，不提供跨檔 rollback。中間 spectrum 集合預設寫到
`<entry>/fluxdep/spectrums.hdf5`。分析不呼叫 Database Save。

## Architecture and owners

分層 `app → Controller(façade) → services → State`。MainWindow 是人使用的 driving
view；RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用
Controller owners。

- [`app.py`](app.py) 只做 controller/window/adapter wiring；程序組合根位於 `apps/fluxdep.py`，`scripts/app/run_fluxdep_gui.py` 只委派 `main(argv)`。
- [`state.py`](state.py)： `FluxDepState`（領域容器）。
- [`analysis_project`](../analysis_project/README.md)：與 dispersive 共用的 entry/component 開啟、qubit alias 預設、kind 驗證、選擇 dialog 與 CLI。
- [`services/`](services/)： 薄包裝純運算，mutate State。
  全部 Qt-free、同步、可獨立測。純運算核心複用
  `zcu_tools.analysis.fluxdep` + `zcu_tools.analysis.spectrum`。
- [`controller.py`](controller.py)： 命令 façade：持 State + EventBus + service，每動作 mutate
  State 後 emit 對應事件。service 保持純（不碰 bus），Controller 是協調層。
  **無 measure 概念**（run/analyze/writeback/context/device/tab）。
- [`interactive.py`](interactive.py)： Qt-free `FluxDepInteractiveOwner` 每張譜保存一份 live 定線／OneTone／TwoTone context 與 Undo，跨譜 Filter 全域一份。GUI controls 與 RPC commands 共用 `gui.interactive` 的 Actions／Session。
  Active 只決定顯示。切換不退休草稿。
  Window close dispose 全部 contexts 並 quiesce runner；view detach 不終止 Session。
- [`event_bus.py`](event_bus.py)： fluxdep 的 payload 型別，掛在共用 `BaseEventBus`
  （`gui/event_bus`，payload-type-key 訂閱）上；bus 機制共用、payload 定義 per-app。
- [`remote/`](remote/README.md)： `RemoteControlAdapter` subclass 共用 `RemoteControlServiceBase`
  （`gui/remote/control_service`），注入 app methods、serializers 與控制身份。

AnalyzePanel 是 **MainWindow 持有的單例**（第一次進 Fit 頁時建立，切頁只隱藏不銷毀），所有階段狀態保留。

[MainWindow](ui/main_window.py) · [AnalyzePanel](ui/analyze_panel.py) · [FilterStage](ui/filter_stage.py) · [互動 widget](ui/interactive/)

## Boundaries and lifetimes

### 領域邊界：不碰 experiment.v2

**不 import `experiment.v2`**（避免把 measure 實驗層拖進來）。

### State 邊界（main-thread 不變式）

所有 State 寫入只在 Qt 主執行緒（沿用 measure 的不變式）。worker（互動 widget 的
背景計算）不直接寫 State，只經 owner-loop delivery 提供計算結果。
TwoTone preview completion 只更新 presentation；Session 與 spectrum points 保留 committed state。

### 繪圖與分析核心

**互動 widget（定線/選點/結果預覽）自持 canvas**。
**v2 search 診斷圖走共用 plot substrate**（`zcu_tools.gui.plotting`，與 measure 共用）：
[search kernel](../../../analysis/fluxdep/README.md) 只算數值；
[診斷圖 builder](../../../plotting/fluxdep/README.md) 回傳原生 Agg Figure。
診斷圖失敗另報 warning，結果與 export 仍可用。

### Remote RPC 與 MCP

共用 remote 擁有 route、owner marshal 與 holder admission，endpoint 擁有 socket、
framing、authentication 與回覆交付。詳見 [remote README](remote/README.md) 與 ADR-0070。

MCP entrypoint 使用共用 bridge，手寫工具組裝基礎 RPC 並摘要資料，control tools 轉送同一
authority。MCP 不訂閱業務 event-push，不維護另一套 holder 或 operation registry。
RPC clients 輪詢 `spectrum.list`、`spectrum.snapshot(name)`、`selection.snapshot` 與
`fit.result` 讀取目前狀態。Search 使用 `fit.search` 回傳的 token 呼叫
`operation.await(token, timeout)`；逾時可再次等待，完成後重讀 `fit.result`。
EventBus 供 app 內部事件使用，不提供業務事件的 RPC 訂閱；controller diagnostics
保留獨立的 broadcast 與 queue 路徑。Agent 不關閉使用者的 GUI。

`Controller.control_identity` 是控制身份的唯一 owner。
Fluxdep 沒有硬體緊急 Stop，使用者必須先點燈收回再取消。

### Database search 的 app ownership

`database.py` 擁有 search HDF5 的 metadata reader，讀 ranges、參數表與 energies 維度，不讀 energy values，也不建立 search cache。這是 fluxdep 專用資料，不屬於通用 datafile 容器。FitService 與 remote dispatch 共用此 reader。


`Controller.search` 的 `FluxDepSearchOwner` 集中 single-flight、版本依賴與 terminal policy。
State 集中定義 alignment → points → Filter → fit 依賴圖，另納入 spectrum 集合。
上游發布保留結果並標過期，`set_fit_params(search_db, bounds, transitions, r_f, sample_f)` 的完整 replacement 則清掉結果。`bounds` 重用 analysis owner 的 `ParamBounds(EJ, EC, EL)`，三組上下限以 GHz 表示。Controller、FitService 與 State 轉交同一組 bounds，不改 RPC 欄位或 UI 控制項。過期 params 匯出在 I/O 前拒絕。
Hide 或 detach 不取消，重新 activate 讀 owner snapshot。
關窗先拒絕新 search 並提出 cancel，search runner 未 drain 就拒絕關窗，保留 Qt owners 與圖。

[Search owner](search.py) · [Fit service](services/fit.py) · [依賴圖與版本](state.py)

### Checkpoints and export

Processed restore 是採點後的 checkpoint，包含零點完成譜。
Export 保存所有已採點，Filter 遮罩留在 Session。
Export 保持 State 原狀。失敗保留已成功的輸出，後續清理由 caller 決定。

[Load service](services/load.py) · [Export service](services/export.py) · [匯出收據](services/fit.py)

## Known limitations

- **軸轉置取決於 Labber step channel 順序**：要看實際檔案。GUI 的「Transpose axes」toggle 讓 user 從 preview 判斷後交換。
- **物理校準驗收**：真實資料的工具契約驗證不等於物理校準驗收。校準是否正確仍需另外判讀分析結果與證據。

## Entry points

- `python -m apps.fluxdep`，啟動（`--control-port` 開 RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server。Agent 流程入口是 [run-fluxdep-gui skill](../../../../../.agents/skills/run-fluxdep-gui/SKILL.md)，工具命令與參數由 MCP descriptions 說明；module README 不擁有操作流程。
