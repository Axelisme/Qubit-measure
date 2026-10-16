# `zcu_tools.gui.app.fluxdep` flux-dependence analysis GUI

**Last updated:** 2026-10-16

MCP server entry 位於 `zcu_tools.mcp.fluxdep.server`；本 package 只包含 GUI app、
state/services/UI 與和 `ui/` 平級的 GUI-process remote driving adapter。Import path 固定為
`zcu_tools.gui.app.fluxdep.*`。

## Module purpose

獨立的分析型 GUI，承接 `notebook_md/analysis/fluxdep_fit.md` 的 fluxonium 能譜擬合
流程。**領域層獨立於 measure-gui**：自己的 state / services /
互動 widget / skill，與 measure-gui 領域零耦合。**共用機制（transport + event）位於
`gui/remote`（`NdjsonRpcEndpoint`、`MethodSpec`）與
`gui/event_bus`（`BaseEventBus`/`BasePayload`），三個 app 共用一份**；domain 仍各
app 自帶（version table、worker 模式等仍 per-app）。

定位是**可選工具**，measure-gui 不含它。

## Pipeline

分析以決策點組織，步驟可反覆執行，順序由人或 agent 決定。資料依賴仍是對齊 → 採樣 → Filter → fit。下圖列出資料流，不是強制操作順序：

```
載入 hdf5 spectrum(OneTone/TwoTone)
  → 定線(LinePicker)       → flux_half / flux_int / flux_period(per-spectrum, 可繼承)
  → 選點(OneTone slider / TwoTone brush mask)  → 每張譜的標註點
  → [累積多張 spectrum 進集合]
  → 跨譜篩選(Selector)     → 每張譜的 Filter 遮罩 + min_distance
  → 匯出 spectrums.hdf5
  → [v2] 搜資料庫(FitPanel) → (EJ,EC,EL) + 視覺化 + 診斷圖 → 匯出 params.json
```

與 measure 的根本差異（為何領域全新寫、不套 adapter/tab/run）：fluxdep 是
**單一 pipeline session**（spectrum 集合），不是平行多 tab；步驟間直接傳資料、
可反覆重做；不碰硬體（無 soc/device/reps/sweep）。

**v2（database search）**：在 selection 之後，用選中的聯合點雲搜 fluxonium 資料庫求
(EJ,EC,EL)，matplotlib 視覺化 + 後端原生診斷圖，匯出 params.json。**只做 search，不做
scipy fit**（fit_spectrum 留在 notebook，未移植）。

## Architecture and owners

分層 `app → Controller(façade) → services → State`。MainWindow 是人使用的 driving
view；RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用
Controller owners。Shared remote 在 owner loop 檢查控制權 holder；讀取只回現況，不建立 per-connection mutation baseline。來源版本與互動 lifetime 仍由 app owner 檢查。

- [`app.py`](app.py) 只做 controller/window/adapter wiring；程序入口位於 `scripts/run_fluxdep_gui.py`。
- [`state.py`](state.py)： `FluxDepState`（領域容器）：`project`(ProjectInfo)、
  `spectrums: dict[str, SpectrumEntry]`、`active_spectrum`、`selection`(SelectionState)、
  `version`(shared VersionTable)。
- [`services/`](services/)： 薄包裝純運算，mutate State：`load`(LoadService)、
  `alignment`(Alignment/Points)、`store`(SpectrumStore/Selection)、`export`。
  全部 Qt-free、同步、可獨立測。純運算核心複用
  `zcu_tools.analysis.fluxdep` + `zcu_tools.analysis.spectrum`。
- [`controller.py`](controller.py)： 命令 façade：持 State + EventBus + service，每動作 mutate
  State 後 emit 對應事件。service 保持純（不碰 bus），Controller 是協調層。
  **無 measure 概念**（run/analyze/writeback/context/device/tab）。
- [`interactive.py`](interactive.py)： Qt-free `FluxDepInteractiveOwner` 每張譜保存一份 live 定線／OneTone／TwoTone context 與 Undo，跨譜 Filter 全域一份。GUI controls 與 RPC commands 共用 `gui.interactive` 的 Actions／Session。
  Active 只決定顯示。切換不退休草稿；同譜換 picker kind 才丟掉該譜草稿，open receipt 回 `discarded`。
  Source reload、remove 與 external publication 退休對應 context，Filter 仍驗證全部來源版本。
  每份 line plugin 有獨立 single-flight auto alignment；共用 background runner 可並行提交不同譜。
  Sampling intent 依已發布譜型開 OneTone 或 TwoTone picker。TwoTone 初次與重新採樣都從空遮罩開始，不繼承已發布點。
  Finish 經 Controller services 發布。Window close dispose 全部 contexts 並 quiesce runner；view detach 不終止 Session。
- [`event_bus.py`](event_bus.py)： fluxdep 的 payload 型別，掛在共用 `BaseEventBus`
  （`gui/event_bus`，payload-type-key 訂閱）上；bus 機制共用、payload 定義 per-app。
- [`remote/`](remote/README.md)： `RemoteControlAdapter` subclass 共用 `RemoteControlServiceBase`
  （`gui/remote/control_service`），注入 app methods、serializers 與控制身份。MCP
  entrypoint 位於 `zcu_tools/mcp/fluxdep/server.py`；`McpBridge` 在
  `zcu_tools/mcp/core/bridge`。

### UI owners

上方列左邊是目前 project 與控制權燈，右邊是 project、加譜、restore 與 export 命令。主區是 Spectra 與 Fit 兩頁 tab：
Spectra 頁是譜清單（Remove 在清單下方）加編輯區，編輯區上方是 `1 Align → 2 Sample` 階段列；Fit 頁承載 AnalyzePanel 單例。
選譜、active 切換與 stage 變更只重建 Spectra 頁的編輯區，不蓋掉 Fit 頁。切頁只跟人類意圖：
使用者點 tab，或 GUI Add／Restore 後回到 Spectra。Agent-origin 的 active、picker、Filter 與
search pending 只更新所屬頁的內容，不把使用者從目前的頁拉走。
離開 Fit 頁只 detach Filter view；進入 Fit 頁經 `show_tab` 重新 attach live Filter，或顯示已發布遮罩。

AnalyzePanel 是 **MainWindow 持有的單例**（第一次進 Fit 頁時建立，切頁只隱藏不銷毀），所有階段狀態保留。

[MainWindow](ui/main_window.py) · [AnalyzePanel](ui/analyze_panel.py) · [FilterStage](ui/filter_stage.py) · [互動 widget](ui/interactive/)

## Boundaries and lifetimes

### 領域邊界：不碰 experiment.v2
LoadService 用底層 `load_data`(datafile) + `format_rawdata`(analysis.spectrum)，
**不 import `experiment.v2`**（避免把 measure 實驗層拖進來）。OneTone/TwoTone
載入完全相同；`spec_type` 只是 metadata，下游選點工具才分支。

### State 邊界（main-thread 不變式）
所有 State 寫入只在 Qt 主執行緒（沿用 measure 的不變式）。worker（互動 widget 的
背景計算）不直接寫 State，只經 owner-loop delivery 提供計算結果。
TwoTone preview completion 只更新 presentation，不提交 Session 或 spectrum points。

### 繪圖與分析核心

**互動 widget（定線/選點/結果預覽）自持 canvas**：widget 自持 `Figure` +
`FigureCanvasQTAgg`，主執行緒 `mpl_connect` 接滑鼠 + 即時 redraw（圖上互動，與
measure plot_host 的單向顯示流方向相反）。`InteractiveMplWidget`(base) 提供 canvas +
可覆寫的 on_press/move/release + 控制項區 + `finished` signal。

**v2 search 診斷圖走共用 plot substrate**（`zcu_tools.gui.plotting`，與 measure 共用）：
[search kernel](../../../analysis/fluxdep/README.md) 只算數值；
[診斷圖 builder](../../../plotting/fluxdep/README.md) 回傳原生 Agg Figure。Qt 在主執行緒記錄搜尋結果後，才建圖、adopt `diagnostic` 並透過 Plots／QtPlotHost 呈現。Panel 持有明確的 owner scheduler 與 presentation 使用期；替換和關窗 release，舊原生圖仍可保存。診斷圖失敗另報 warning，結果與 export 仍可用。Notebook caller 以 IPython display 發布普通圖。

[fluxdep kernel README](../../../analysis/fluxdep/README.md) 下，互動選點、filtering、line selection、one-tone peak detection 的共用規則住在
`zcu_tools.analysis.fluxdep`。Qt `ui/interactive/` widget 只保留控制項、canvas、worker/debounce
與 Qt event translation；共用躍遷換算與 database search 位於 `analysis.fluxdep`，診斷圖由 `plotting.fluxdep` 建立，params export 留在 app pipeline。

### flux 對齊：per-spectrum + 可繼承
每張譜各自一份 flux_half/int/period（對齊 analysis.fluxdep.models.SpectrumResult）。新載入的譜可
`inherit_from` 既有譜的對齊當初值（`alignment_seeded` 標記），LinePicker 才會 seed；
fresh load 用 picker 預設。OneTone 譜的 LinePicker 鎖 magnitude-only（相位無資訊）。
`inherited_from` 表示目前已發布對齊的來源。未改繼承線直接 Finish 保留來源；
手動改線或 Auto Align 後 Finish 才清除。丟棄草稿不改已發布對齊或來源，
Controller 直接發布手動對齊則清除來源。

### 跨譜篩選與 Session

Apply 與 Finish 一起發布 kept 遮罩、這次 Session 的刷選與 min_distance。App owner 建立新的跨譜
context 時，以已發布的刷選為初始選取（`State.brush_mask()`，沒有遮罩的譜全選），並繼承已發布的
`SelectionState.min_distance`；沒有任何已發布遮罩時全選。GUI 的 Edit 與 RPC
`selection.interactive.open` 走同一個 owner 入口。同刷選、同 min_distance 原樣 Finish，結果等於已發布的
kept 遮罩；改 min_distance 時從刷選重算。Restore 回來的譜沒有刷選，以全部點當刷選，
所以 min_distance 大於 0 時，只有這種譜重開後原樣 Finish 可能再刪點。

Holder 按 Edit（`fluxdep.filter.edit`）開 Session，按 Discard（`fluxdep.filter.discard`）取消；Selector 的 Finish
發布並關閉，Apply 發布但不關閉。離開 Filter 階段或 Fit 頁一律只 detach view，回到 Filter 時重新 attach 同一個 Session；
Session 留到 Finish、Discard 或上游改動使它失效。

### Remote RPC 與 MCP

`RemoteControlAdapter` 注入 app methods、event serializers 與 application control identity。
共用 remote 擁有 route、owner marshal 與 holder admission，endpoint 擁有 socket、
framing、authentication 與回覆交付。Read 不建立 seen baseline，寫入預設要求 holder；
readonly method 才允許 observer。詳見 [remote README](remote/README.md) 與 ADR-0070。

MCP entrypoint 使用共用 bridge，手寫工具組裝基礎 RPC 並摘要資料，control tools 轉送同一
authority。MCP 不訂閱業務 event-push，不維護另一套 holder 或 operation registry。
RPC clients 可訂閱 EventBus facts。Agent 不關閉使用者的 GUI。

### Qt 控制身份與編輯停用

`Controller.control_identity` 是控制身份的唯一 owner。共用 `ControlIndicator` 顯示
Local、Local 鎖定及 Remote label／在線狀態。點燈呼叫 `toggle_local`，收回成 Local
鎖定或切換 Local 鎖定；轉移本身不取消 search 或 interactive Session。

Fluxdep 沒有硬體緊急 Stop，使用者必須先點燈收回再取消。

Remote 關窗先確認，拒絕不開始 shutdown。接受後沿原 search cancel／drain 與
interactive dispose 契約處理，不自動收回控制權。Close 釋放主視窗的 authority watch。背景 drain、app-owned Session 與 EventBus
subscriptions 保留原 lifetime。

### Database search 的 app ownership
`FitService.capture_search` 在 State owner capture detached `SearchInput`。
`compute_search(inputs)` 只算數值，不讀 live State、不繪圖。
`record_result(result, inputs)` 在 owner 寫入同次 `SearchInput` 的原始 normalized-flux／GHz 點雲、點序與版本，Controller 發布 `FitChanged`。Kernel 的 modulo-one flux 不取代背景上的原始座標。
State 集中定義 alignment → points → Filter → fit 依賴圖，另納入 spectrum 集合。Search capture 包含所有來源，零點譜也在內。Search delivery、結果過期原因與 publication invalidated 共用此圖。
上游發布保留結果並標過期，`set_fit_params` 的完整 replacement 則清掉結果。過期 params 匯出在 I/O 前拒絕。

`Controller.search` 的 `FluxDepSearchOwner` 集中 single-flight、版本依賴與 terminal policy。
依賴包含 project、fit、selection、spectrum 集合與全部來源，包含零點譜。
來源改變時，成功 delivery 回 failed，不覆蓋新的 fit。Active spectrum 切換不影響 joint search。
Cancel 只提出請求，kernel 的 `SearchCancelled` 才代表運算取消。
普通失敗保持 failed，最後 checkpoint 後的有效成功可以 finished。

App composition 注入專用 `BackgroundRunner` 與 `ProgressService`，沒有 hardware gate。
AnalyzePanel 從同一 owner 與 progress facet 顯示 Search／Cancel／結果。
Hide 或 detach 不取消，重新 activate 讀 owner snapshot。
關窗先拒絕新 search 並提出 cancel，search runner 未 drain 就拒絕關窗，保留 Qt owners 與圖。
診斷圖失敗不改數值 outcome。

`Controller.search_database` 保留 headless owner-inline capture／compute／record 便利入口，
不取 operation token。Remote search 使用同一 search owner，不另建 operation registry。

[Search owner](search.py) · [Fit service](services/fit.py) · [依賴圖與版本](state.py)

### Caller-correctable errors

State、SearchOwner 與 load／fit／export owners 使用 shared `InvalidInputError` 與
`FailedPreconditionError` 分類 caller 可修正的失敗。State 的 `get_spectrum` 是 owner-thread
literal-name lookup，回傳 live entry；未知名稱不修改來源或版本。Remote 只投影 nominal category
與 reason，不從 ordinary exception ancestry 或訊息猜分類。Missing runtime、thread misuse、
provider I/O 與 worker failure 保留各自的 unexpected failure／operation outcome 語意。

### Checkpoints and export

Processed restore 是採點後的 checkpoint，包含零點完成譜。同名衝突在發布前拒絕整批，列出衝突名稱。
Export 保存所有已採點，不保存 Filter 遮罩。其他 publication 失敗保留已發布前綴，不 rollback。

Spectrum export 預設 create-only，明示 overwrite 才替換。Params export 沿原生 merge，
保留 sections，取第一張 aligned spectrum 的 calibration，包含零點譜。
Export 不改 State；失敗保留成功前綴，不 cleanup／rollback。

[Load service](services/load.py) · [Export service](services/export.py) · [匯出收據](services/fit.py)

## Known limitations

- **spec_type 持久化**：`dump_spectrums`/`load_spectrums` 把 type 存成 h5 group
  attribute；舊檔無 attr 則 type 不設、restore 時 fallback TwoTone。
- **軸轉置取決於 Labber step channel 順序**：`load_data` 寫死把 `data[:,0,0]` 當 dev、
  `data[0,1,:]` 當 freq，即假設 step channel 是 `[Flux, Frequency]`。但 OneTone 量測常存成
  `[Frequency, Flux]`（freq 掃在外層）→ 軸反。**不是固定特性**（TwoTone 通常正、OneTone 常反），
  要看實際檔案。GUI 的「Transpose axes」toggle（`services/load.py` 的 `transpose_spectrum_data`）
  讓 user 從 preview 判斷後交換。
- **物理校準驗收**：真實資料的工具契約驗證不等於物理校準驗收。校準是否正確仍需另外判讀分析結果與證據。

## Entry points

- `scripts/run_fluxdep_gui.py`，啟動（`--control-port` 開 RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server。Agent 流程入口是 [run-fluxdep-gui skill](../../../../../.agents/skills/run-fluxdep-gui/SKILL.md)，工具命令與參數由 MCP descriptions 說明；module README 不擁有操作流程。
