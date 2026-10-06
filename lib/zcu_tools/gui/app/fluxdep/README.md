**Last updated:** 2026-10-06. Project RPC and shared resource observations

# `zcu_tools.gui.app.fluxdep` — flux-dependence analysis GUI

MCP server entry 位於 `zcu_tools.mcp.fluxdep.server`；本 package 只包含 GUI app、
state/services/UI 與和 `ui/` 平級的 GUI-process remote driving adapter。Import path 固定為
`zcu_tools.gui.app.fluxdep.*`。

## Module Purpose

獨立的分析型 GUI，承接 `notebook_md/analysis/fluxdep_fit.md` 的 fluxonium 能譜擬合
流程。**領域層獨立於 measure-gui**：自己的 state / services /
互動 widget / skill，與 measure-gui 領域零耦合。**共用機制（transport + event）位於
`gui/remote`（`NdjsonRpcEndpoint`、`MethodSpec`）與
`gui/event_bus`（`BaseEventBus`/`BasePayload`），三個 app 共用一份**；domain 仍各
app 自帶（version table、worker 模式等仍 per-app）。

定位是**可選工具**，measure-gui 不含它。

## Pipeline

一條線性有序、人在迴圈中校驗、可回退重做的 pipeline：

```
載入 hdf5 spectrum(OneTone/TwoTone)
  → 定線(LinePicker)       → flux_half / flux_int / flux_period(per-spectrum, 可繼承)
  → 選點(OneTone slider / TwoTone brush mask)  → 每張譜的標註點
  → [累積多張 spectrum 進集合]
  → 跨譜篩選(Selector)     → 聯合點雲的 selected 遮罩 + min_distance
  → 匯出 spectrums.hdf5
  → [v2] 搜資料庫(FitPanel) → (EJ,EC,EL) + 視覺化 + 診斷圖 → 匯出 params.json
```

與 measure 的根本差異（為何領域全新寫、不套 adapter/tab/run）：fluxdep 是
**單一 pipeline session**（spectrum 集合），不是平行多 tab；步驟間直接傳資料、
可反覆重做；不碰硬體（無 soc/device/reps/sweep）。

**v2（database search）**：在 selection 之後，用選中的聯合點雲搜 fluxonium 資料庫求
(EJ,EC,EL)，matplotlib 視覺化 + 後端原生診斷圖，匯出 params.json。**只做 search，不做
scipy fit**（fit_spectrum 留在 notebook，未移植）。

## Architecture Overview

分層 `app → Controller(façade) → services → State`。MainWindow 是人使用的 driving
view；RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用
Controller owners，資源的 per-connection observation 與 guard 由 shared remote 執行。
目前支援完整 project／spectrum／selection 查詢及 project／spectrum 編輯命令。
載入、search、export 與 interactive RPC 正在 task 中實作。

- **`state.py`** — `FluxDepState`（領域容器）：`project`(ProjectInfo)、
  `spectrums: dict[str, SpectrumEntry]`、`active_spectrum`、`selection`(SelectionState)、
  `version`(shared VersionTable)。Spectrum leaf 移除使用精確 retire，刪除期間版本為 0，
  同名重建續增，不重用舊 observation；不影響同字首的其他譜。Collection／global keys
  保持 State lifetime 的計數。
  **`ProjectInfo`/`default_*` 共用** `gui/project.py`（Qt-free，與 dispersive 同源）；
  `ProjectDialog` 共用 `gui/widgets/project_dialog.py`（`db_label="Database path"`），並可從
  project root 掃描到的 `result/**/params.json` result scope 下拉選取既有 chip/qubit。
  `SpectrumEntry` 持 raw(SpectrumData)/points(PointsData)/per-spectrum flux 對齊/
  aligned/points_completed/alignment_seeded。raw/points 直接複用
  `analysis.spectrum` 與 `analysis.fluxdep.models` 的 **TypedDict**（欄位用 `[...]` 存取，非 dataclass）。
- **`services/`** — 薄包裝純運算，mutate State：`load`(LoadService)、
  `alignment`(Alignment/Points)、`store`(SpectrumStore/Selection)、`export`。
  全部 Qt-free、同步、可獨立測。純運算核心複用
  `zcu_tools.analysis.fluxdep` + `zcu_tools.analysis.spectrum`。
- **`controller.py`** — 命令 façade：持 State + EventBus + service，每動作 mutate
  State 後 emit 對應事件。service 保持純（不碰 bus），Controller 是協調層。
  **繼承共用 `BaseController`**（`gui/controller_base`，generic over State+Bus）取得
  state/bus/project_root 儲存 + `state`/`bus` property + `get_project_root` + `_emit`
  helper；per-command façade body 仍各 app（領域動詞 + app payload），main 不繼承。
  **無 measure 概念**（run/analyze/writeback/context/device/tab）。
- **`interactive.py`** — Qt-free `FluxDepInteractiveOwner` 持有 active spectrum 的
  單一 live 定線／OneTone／TwoTone／跨譜篩選 context。GUI controls 與 commands 共用 `gui.interactive` 的 Actions／Session；
  LinePicker 只持有 disposable preview，OneTone threshold 與 indices 一起 commit／undo。
  TwoTone Session 保存 mask、detector 與 tools；工具設定保留單層 Undo，完整 stroke 一次 commit。
  Background projection 只提供 derived view；Finish 重新計算 committed snapshot，不等待 preview。
  Finish 經 Controller 的 AlignmentService／PointsService 發布；後者擁有排序與 flux calibration。
  Picker kind switch、active switch、reload、remove 或 external spectrum change 關閉舊輸入。
  跨譜 context capture 全部來源版本，包含零點譜；任何來源變更或外部 selection publication 都關閉舊輸入。
  Apply 同步計算最新 snapshot、經 Controller 發布一次 selection，保留 input 與 Undo，不是 picking Finish。
  Widget detach 不終止 session，window close 才 dispose owner 並 quiesce 背景 runner。
- **`event_bus.py`** — fluxdep 的 payload 型別，掛在共用 `BaseEventBus`
  （`gui/event_bus`，payload-type-key 訂閱）上；bus 機制共用、payload 定義 per-app。
- **`ui/`** — `MainWindow`（左 spectrum 列表 + 右階段驅動編輯區）、互動 widget
  （`ui/interactive/`）、`load_dialog`/`export_dialog`。共用層（與 dispersive）：
  `LoadSpectrumDialog` subclass `gui/widgets/load_dialog.LoadDataDialog`（共用 file
  row/transpose/preview/OK-gate，`_build_options` 掛 Type/Inherit combo、`result_request`
  回 `LoadRequest`）；`error_messages` 用共用 `gui/error_messages` framework（domain rule
  各 app）；`ui/paths.nearest_existing` 來自 `gui/project`；`ui/interactive/display.contrast_limits`
  一份供 find_points/result_preview。MainWindow 擁有的 EventBus subscriptions 在 window close
  釋放，避免分析 view 被 bus callback 保活。
- **`remote/`** — `RemoteControlAdapter` subclass 共用 `RemoteControlServiceBase`
  （`gui/remote/control_service`），注入 app observation policies 與 resource versions。MCP
  entrypoint 位於 `zcu_tools/mcp/fluxdep/server.py`；`McpBridge` 在
  `zcu_tools/mcp/core/bridge`。

## Key Design Decisions

### 領域邊界：不碰 experiment.v2
LoadService 用底層 `load_data`(datafile) + `format_rawdata`(analysis.spectrum)，
**不 import `experiment.v2`**（避免把 measure 實驗層拖進來）。OneTone/TwoTone
載入完全相同；`spec_type` 只是 metadata，下游選點工具才分支。

### State 邊界（main-thread 不變式）
所有 State 寫入只在 Qt 主執行緒（沿用 measure 的不變式）。worker（互動 widget 的
背景計算）不直接寫 State，只經 owner-loop delivery 提供計算結果。
TwoTone preview completion 只更新 presentation，不提交 Session 或 spectrum points。

### 兩種繪圖機制：互動 widget 自建 canvas / v2 診斷圖使用 explicit host
**互動 widget（定線/選點/結果預覽）自持 canvas**：widget 自持 `Figure` +
`FigureCanvasQTAgg`，主執行緒 `mpl_connect` 接滑鼠 + 即時 redraw（圖上互動，與
measure plot_host 的單向顯示流方向相反）。`InteractiveMplWidget`(base) 提供 canvas +
可覆寫的 on_press/move/release + 控制項區 + `finished` signal。

**v2 search 診斷圖走共用 plot substrate**（`zcu_tools.gui.plotting`，與 measure 共用）：
[search kernel](../../../analysis/fluxdep/README.md) 只算數值；
[診斷圖 builder](../../../plotting/fluxdep/README.md) 回傳原生 Agg Figure。Qt 在主執行緒記錄搜尋結果後，才建圖、adopt `diagnostic` 並透過 Plots／QtPlotHost 呈現。Panel 持有明確的 owner scheduler 與 presentation 使用期；替換和關窗 release，舊原生圖仍可保存。診斷圖失敗另報 warning，結果與 export 仍可用。Notebook caller 以 IPython display 發布普通圖。
共用套件分工：
- `plotting/host.py` 提供 explicit attach／remove 的主執行緒 bridge 與 figure registry。
- `runtime.py` 在 behavior 建立前設定 logging，在 QApplication 建立後初始化
  host、shutdown callback 與 mathtext 支援，不切換全域 Matplotlib backend。
- `FluxDepGuiBehavior.spec` 宣告 app slug 與 default control port。`app.py` 只做
  controller/window/adapter wiring；程序入口位於 `scripts/run_fluxdep_gui.py`。
- DB 搜尋由 app-owned `FluxDepSearchOwner` 提交到專用 `BackgroundRunner`。
  Worker 只計算 detached snapshot。Owner 提交有效數值後，panel 才呈現診斷圖。

### 編輯區階段驅動
MainWindow 編輯區依 active 譜的 pipeline 階段 swap widget：未定線→LinePicker；
已定線未完成選點→OneTone/FindPoints(by spec_type)；已完成→ResultPreview(唯讀結果圖)。
空 Finish 也完成選點，清單顯示完成，ResultPreview 顯示零點。
`points_completed` 表示階段完成；`point_count` 表示可用點數。Analyze／Selector 只使用非空資料。
widget 的 `finished` → Controller 寫回 → 階段前進 → 重 swap。
ResultPreview 的 Re-select points 透過 Controller 清空 points／completion 並重開 selector。
Re-pick lines 重開 alignment，保留 native points／completion；接受新定線時一次重算 raw 與 point fluxs。
Processed restore 包含零點的完成結果，不新增持久化欄位。

### 背景計算 + 即時中斷（generation 戳記）
慢計算經共用 `BackgroundRunner.submit`（per-panel，`enter=None`）off-main，避免拖動卡 UI。
**用 generation 計數即時中斷**：參數變遞增 generation + debounce(80ms) 啟 worker；
`on_done` 帶 captured generation，主執行緒檢查不是最新就丟棄（非中途 kill）。`get_result`
同步算最終（finish 終點）。**generation/debounce 留在 panel、不進 runner**——這個「最新者勝」
取消範式與 measure 的 stop_event 協作取消不同，刻意不合併（runner 對取消無感）。
- **LinePicker** 使用共用 plugin 的 single-flight auto alignment；owner-loop completion
  對最新 snapshot 提交，已終止 session 的晚到結果不發布。它不使用 panel generation 範式。
- **FindPoints** `spectrum2d_findpoint`（大譜 ~180-480ms/次）：worker 化。
- **Selector** `downsample_points`（O(N²)，5000 點 ~1.3s）：worker 化。
- **線程非進程**：實測 numpy/scipy 釋放 GIL，背景線程不卡主執行緒；避開進程的
  大 signals 陣列 IPC 序列化開銷。
- **OneTone** `find_peaks` 只 0.1ms，**不 worker 化**；最大色散頻率與 inverted slice
  預處理使用 `zcu_tools.analysis.fluxdep` one-tone kernel。它的拖動卡是 redraw 整張 figure，對症優化是
  **重用 scatter(set_offsets) + debounce redraw(50ms)**。

### Flux-Dependence Analysis kernel handoff
[fluxdep kernel README](../../../analysis/fluxdep/README.md) 下，互動選點、filtering、line selection、one-tone peak detection 的共用規則住在
`zcu_tools.analysis.fluxdep`。Qt `ui/interactive/` widget 只保留控制項、canvas、worker/debounce
與 Qt event translation；共用躍遷換算與 database search 位於 `analysis.fluxdep`，診斷圖由 `plotting.fluxdep` 建立，params export 留在 app pipeline。

### flux 對齊：per-spectrum + 可繼承
每張譜各自一份 flux_half/int/period（對齊 analysis.fluxdep.models.SpectrumResult）。新載入的譜可
`inherit_from` 既有譜的對齊當初值（`alignment_seeded` 標記），LinePicker 才會 seed；
fresh load 用 picker 預設。OneTone 譜的 LinePicker 鎖 magnitude-only（相位無資訊）。

### Interactive context facts 與 GUI 跟隨
Owner 的 inspect 回傳 live context reference 與 owner-lifetime identity，不建立 Session，也不消耗 Undo。
同一有效 begin 重用 identity；退休後的新 context 使用新 identity。Identity 不代表 edit revision，
同一 Session 的 GUI／command Actions 仍依 owner-loop 順序提交。

Owner 發布 opened／updated／closed domain facts，Session commit／undo 與定線 alignment info 都更新同一 context。
MainWindow 依 EventMeta origin 與目前 identity 顯示 agent 開啟／修改的 picker 或 Filter。
GUI 依 context kind 掛載，不能用已完成的 pipeline 階段代替目前 context。
Closed 只清除旧 view，不開新 Session。純讀取與 user-origin edits 不強制切換畫面。
AnalyzePanel 的 public show_tab 管理 Filter／Search／Show，外部不存取 Qt 私有 stack 或 tab。
Agent-origin search pending 顯示 lazy Search 面板；terminal 更新結果，不搶回使用者已切換的畫面。

### 跨譜篩選：繼承 min_distance 不繼承 select
App owner 建立新的跨譜 context 時全選，僅繼承已發布的 `SelectionState.min_distance`。
Analyze singleton 每次啟用 Filter 都重新 attach 有效 context；離開 Filter 取消 context，切走 Analyze 則只 detach view。
Selector controls／完整 stroke 透過共用 Actions 修改 Session。Preview 使用 80ms debounce 與 generation guard；
Apply 不依賴 preview worker，teardown 阻擋 hidden 舊 controls 與晚到 delivery。

### 配色
互動圖背景一律 `gray_r`（白底、高值=黑），紅點落在高值共振線上對比最強
（vs viridis 高值=黃，紅點不明顯）。對齊 notebook plotly 版的 Greys。

### Remote RPC 與 MCP

`RemoteControlAdapter` 注入 app 的 methods、event serializers、resource versions 與 observation policies。
`RemoteControlServiceBase` 擁有 route、owner marshal、每條連線的 seen map 與 guard。
`NdjsonRpcEndpoint` 擁有 socket、framing、authentication 與回覆交付。App 不複製這些機制。

完整 `project.info`、`spectrum.list`、`fit.result` 分別揭露 project、集合及 fit。
`selection.pointcloud`、`state.check` 與 `resources.versions` 不建立完整 observation。
`project.setup` 必須先讀 project，使用 Controller 與原生 ProjectInfo paths。成功 self-write
只推進同連線已讀且相符的版本，其他連線與 GUI 修改後仍須明示重讀。
Spectrum snapshot 回完整 calibration／published points 與 raw axes extents，不輸出 complex signal matrix。
合法的 absent name 回 absence 並建立 version 0 observation。Leaf guard 不展開 name 內的 literal star。
Remove、reset 與 active selection 共用原生 Controller owners；同名重建會使舊 observation stale。
Selection snapshot 回完整 published mask 與 normalized min_distance，不讀 live Session。
六個原有 read projections 保持原值；其餘 pipeline methods 尚在實作。

MCP entrypoint 使用共用 `McpBridge`，工具從 method specs 生成。完整控制工具的 workflow
與圖像驗收尚未完成。MCP 不訂閱業務 event-push，不維護第二份 seen map。
RPC clients 可自行訂閱 EventBus facts。Agent 不關閉使用者的 GUI。

### Database search 的 app ownership
`FitService.capture_search` 在 State owner capture detached `SearchInput`。
`compute_search(inputs)` 只算數值，不讀 live State、不繪圖。
`record_result` 在 owner 寫入 fit，Controller 發布 `FitChanged`。

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
不取 operation token。Remote search 尚在實作，使用同一 search owner，不另建 operation registry。

### Caller-correctable errors

State、SearchOwner 與 load／fit／export owners 使用 shared `InvalidInputError` 與
`FailedPreconditionError` 分類 caller 可修正的失敗。State 的 `get_spectrum` 是 owner-thread
literal-name lookup，回傳 live entry；未知名稱不修改來源或版本。Remote 只投影 nominal category
與 reason，不從 ordinary exception ancestry 或訊息猜分類。Missing runtime、thread misuse、
provider I/O 與 worker failure 保留各自的 unexpected failure／operation outcome 語意。

### v2 結果存放 + 視覺化
- `FitState`（State 上的 singleton，version key `fit`）：db 路徑/EJb/ECb/ELb/transitions/r_f/sample_f
  + 結果 params(EJ,EC,EL)。`set_fit_params` 改輸入會清掉舊結果（輸入變則舊結果失效）。
- **AnalyzePanel UI**（`ui/analyze_panel.py`）：selection 後的三步分析集中到**一個「Analyze…」按鈕**
  開的單例面板，內含 **QTabWidget: Filter / Search / Show**（取代舊的 Cross-spectrum filter + Fit
  spectrum 兩按鈕）。
  - **Filter**：嵌 cross-spectrum `SelectorWidget`（進此 tab 時依當前 spectrums 重建）。
  - **Search**：db 搜尋表單（bounds preset 下拉 `general`/`integer`/`all` 填三組 bound spinbox——
    **preset 綁 bounds 不綁 transitions**，transitions 是 `TransitionsForm` 純手填、無自己的 preset）+
    右側診斷圖（QSplitter 可拖拉）。search 前擋空/不存在 db 路徑；params export 的長路徑只作
    status/tooltip，不參與 panel 水平最小寬度。
  - **Show**：fit 視覺化 + 顯示工具：x/y 軸上下限數字框（預設按 `viz.derive_auto_limits` = notebook
    `auto_derive_limits`）、r_f/sample_f 參考線 checkbox、要顯示的 transitions 子集（獨立於 fit 用的）。
  AnalyzePanel 是 **MainWindow 持有的單例**（建一次留 stack，切走只隱藏不銷毀），所有 tab 狀態保留。
- Search 診斷图 builder 建立原生 Agg Figure，不登記 pyplot manager。Panel 替換或關閉時 release presentation，不以全域 `plt.close("all")` 管理其他 caller 的圖。
- `transitions` 沿用 `analysis.fluxdep.models.TransitionDict`（TypedDict + extra_items，混合 r_f/sample_f scalar
  與任意 `transitions{n}`/`mirror{n}` 動態 list 群）——這正是 extra_items 的設計用途，**不改 pydantic/
  dataclass**（會更弱型）。
- `services/viz.py`：matplotlib 重寫 notebook 的 plotly `FreqFluxDependVisualizer`，純函式畫進傳入的
  Figure（background heatmap gray_r + simulation lines + 選中點 + r_f/sample_f const-freq 線 +
  dev_value secondary axis）。診斷圖直接用共用 builder 的後端原生 Figure，不重畫。
- params.json 的 flux_half/int/period 取**第一張已對齊譜**（notebook 單譜語意；多譜同對齊到同 flux 座標）。
- params.json export 透過 `resources.qubit_params.QubitParams` 寫 `project` 與 `fluxdep_fit`；重寫 fluxdep fit 會更新 `fluxdep_fit.timestamp`，但不刪除獨立的 `dispersive` section。

## Known Limitations

- **spec_type 持久化**：`dump_spectrums`/`load_spectrums` 把 type 存成 h5 group
  attribute；舊檔無 attr 則 type 不設、restore 時 fallback TwoTone。
- **軸轉置取決於 Labber step channel 順序**：`load_data` 寫死把 `data[:,0,0]` 當 dev、
  `data[0,1,:]` 當 freq，即假設 step channel 是 `[Flux, Frequency]`。但 OneTone 量測常存成
  `[Frequency, Flux]`（freq 掃在外層）→ 軸反。**不是固定特性**（TwoTone 通常正、OneTone 常反），
  要看實際檔案。GUI 的「Transpose axes」toggle（`services/load.py` 的 `transpose_spectrum_data`）
  讓 user 從 preview 判斷後交換。
- **Remote pipeline 尚未完成**：目前 agent 可觀測與編輯 project／spectrum，載入、search、export
  與互動命令仍待實作。完整 MCP workflow 與圖像驗收由 fluxdep-mcp-control task 推進。

## Entry Points

- `scripts/run_fluxdep_gui.py` — 啟動（`--control-port` 開 RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server；skill `run-fluxdep-gui`
  (`.claude/skills/`，三副本同步 .agent/.codex；`sync_skills.sh` 只同步 SKILL.md) 只含
  SKILL.md。既有 skill 尚描述 read-only workflow；完整控制 workflow 與 skill 更新留在 MCP tools task。
