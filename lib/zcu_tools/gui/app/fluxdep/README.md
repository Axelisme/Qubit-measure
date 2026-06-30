**Last updated:** 2026-07-01

# `zcu_tools.gui.app.fluxdep` — flux-dependence analysis GUI

MCP server entry 位於 `zcu_tools.mcp.fluxdep.server`；本 package 只包含 GUI app、
state/services/UI 與 GUI-process remote adapter。Import path 固定為
`zcu_tools.gui.app.fluxdep.*`。

## Module Purpose

獨立的分析型 GUI，把 `notebook_md/analysis/fluxdep_fit.md` 的 fluxonium 能譜擬合
流程移植成 Qt 桌面工具。**領域層獨立於 measure-gui**：自己的 state / services /
互動 widget / skill，與 measure-gui 領域零耦合。**共用機制（transport + event）已
上提到 `gui/remote`（`NdjsonRpcEndpoint`、`MethodSpec`）與
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

分層 `app → Controller(façade) → services → State`。MainWindow 是唯一的 driving
view（給人）；RemoteControlAdapter 是 **read-only observing view**（給 agent 讀
狀態，不驅動分析）。仿 measure ADR-0013 的 view-split 機制，但 fluxdep 的 remote
view 只暴露查詢，不暴露 mutation。

- **`state.py`** — `FluxDepState`（領域容器）：`project`(ProjectInfo)、
  `spectrums: dict[str, SpectrumEntry]`、`active_spectrum`、`selection`(SelectionState)、
  `version`(VersionTable)。`VersionTable` 原樣搬自 measure（純樂觀鎖機制）。
  **`ProjectInfo`/`default_*` 共用** `gui/project.py`（Qt-free，與 dispersive 同源）；
  `ProjectDialog` 共用 `gui/widgets/project_dialog.py`（`db_label="Database path"`）。
  `SpectrumEntry` 持 raw(SpectrumData)/points(PointsData)/per-spectrum flux 對齊/
  aligned/points_selected/alignment_seeded。raw/points 直接複用
  `notebook.persistance` 的 **TypedDict**（欄位用 `[...]` 存取，非 dataclass）。
- **`services/`** — 薄包裝純運算，mutate State：`load`(LoadService)、
  `alignment`(Alignment/Points)、`store`(SpectrumStore/Selection)、`export`。
  全部 Qt-free、同步、可獨立測。純運算核心複用
  `zcu_tools.analysis.fluxdep` + `notebook.persistance`。
- **`controller.py`** — 命令 façade：持 State + EventBus + service，每動作 mutate
  State 後 emit 對應事件。service 保持純（不碰 bus），Controller 是協調層。
  **繼承共用 `BaseController`**（`gui/controller_base`，generic over State+Bus）取得
  state/bus/project_root 儲存 + `state`/`bus` property + `get_project_root` + `_emit`
  helper；per-command façade body 仍各 app（領域動詞 + app payload），main 不繼承。
  **無 measure 概念**（run/analyze/writeback/context/device/tab）。
- **`event_bus.py`** — fluxdep 的 payload 型別，掛在共用 `BaseEventBus`
  （`gui/event_bus`，payload-type-key 訂閱）上；bus 機制共用、payload 定義 per-app。
- **`ui/`** — `MainWindow`（左 spectrum 列表 + 右階段驅動編輯區）、互動 widget
  （`ui/interactive/`）、`load_dialog`/`export_dialog`。共用層（與 dispersive）：
  `LoadSpectrumDialog` subclass `gui/widgets/load_dialog.LoadDataDialog`（共用 file
  row/transpose/preview/OK-gate，`_build_options` 掛 Type/Inherit combo、`result_request`
  回 `LoadRequest`）；`error_messages` 用共用 `gui/error_messages` framework（domain rule
  各 app）；`ui/paths.nearest_existing` 來自 `gui/project`；`ui/interactive/display.contrast_limits`
  一份供 find_points/result_preview。
- **`services/remote/`** — `RemoteControlAdapter` subclass 共用 `RemoteControlServiceBase`
  （`gui/remote/control_service`，零 policy 覆寫），讓 agent **只讀**觀測（無任何 mutation RPC）。MCP
  entrypoint 位於 `zcu_tools/mcp/fluxdep/server.py`；`McpBridge` 在
  `zcu_tools/mcp/core/bridge`。

## Key Design Decisions

### 領域邊界：不碰 experiment.v2
LoadService 用底層 `load_data`(utils/datasaver) + `format_rawdata`(persistance)，
**不 import `experiment.v2`**（避免把 measure 實驗層拖進來）。OneTone/TwoTone
載入完全相同；`spec_type` 只是 metadata，下游選點工具才分支。

### State 邊界（main-thread 不變式）
所有 State 寫入只在 Qt 主執行緒（沿用 measure 的不變式）。worker（互動 widget 的
背景計算）不直接寫 State，只 emit Qt signal → 主執行緒 slot 寫。

### 兩種繪圖機制：互動 widget 自建 canvas / v2 診斷圖走 plot_host backend
**互動 widget（定線/選點/結果預覽）自持 canvas**：widget 自持 `Figure` +
`FigureCanvasQTAgg`，主執行緒 `mpl_connect` 接滑鼠 + 即時 redraw（圖上互動，與
measure plot_host 的單向顯示流方向相反）。`InteractiveMplWidget`(base) 提供 canvas +
可覆寫的 on_press/move/release + 控制項區 + `finished` signal。

**v2 search 診斷圖走共用 plot substrate**（`zcu_tools.gui.plotting`，與 measure 共用）：notebook 的
`search_in_database(plot=True)` 內部用 pyplot（`plt.figure()`/`plt.show()`）——要在 worker
跑且**不改 fitting.py**，就靠攔截 pyplot 路由內嵌。共用套件:
- `plotting/backend.py`（client）：`module://zcu_tools.gui.plotting.backend`，攔 `plt.figure()` →
  attach 到當前 `FigureContainer`；`plt.show()` → activate（**未 attach 則 raise**，Fast-Fail 統一）；
  `GuiFigureCanvas.draw_idle` 吃跨線程。
- `plotting/host.py`：單一主線程 bridge QObject（訊號在 host）+ figure registry + lifecycle。
- `plotting/routing.py`：task-local `ContextVar`（共用版統一用此，fluxdep 單槽是退化用法）。
- `plotting/setup.py`：`configure_matplotlib_backend()`（pyplot import 前呼，import-clean）；
  `run_fluxdep_gui.py` 入口呼叫；QApplication/`ensure_host()`/`aboutToQuit→set_shutting_down(True)`/建窗/起 adapter 這段 bootstrap 已抽共用 `run_qt_app`（`gui/run_app`），`app.py` 的 `run_app` 塌成 factory 接線（controller/window/adapter factory）。
- **FitPanel R4**：DB 搜尋經共用 `gui/background.py` `BackgroundRunner`（per-panel）提交，
  `enter=` CM 組合 `routing_scope(diag_container)` + `use_pbar_factory(factory)`，由 runner
  在 worker 執行緒**於 thunk 內**進入（ContextVar 在 QThreadPool worker 裡看不到主執行緒的
  `.set()`，故必須在 worker 端設）。這修了原本 worker 裡 plt.show() 崩潰 + 圖彈獨立視窗的 bug。

### 編輯區階段驅動
MainWindow 編輯區依 active 譜的 pipeline 階段 swap widget：未定線→LinePicker；
已定線未選點→OneTone/FindPoints(by spec_type)；已選點→ResultPreview(唯讀結果圖)。
widget 的 `finished` → Controller 寫回 → 階段前進 → 重 swap。
ResultPreview 內含 Re-pick lines / Re-select points 按鈕，可回退任一階段重做。

### 背景計算 + 即時中斷（generation 戳記）
慢計算經共用 `BackgroundRunner.submit`（per-panel，`enter=None`）off-main，避免拖動卡 UI。
**用 generation 計數即時中斷**：參數變遞增 generation + debounce(80ms) 啟 worker；
`on_done` 帶 captured generation，主執行緒檢查不是最新就丟棄（非中途 kill）。`get_result`
同步算最終（finish 終點）。**generation/debounce 留在 panel、不進 runner**——這個「最新者勝」
取消範式與 measure 的 stop_event 協作取消不同，刻意不合併（runner 對取消無感）。
- **FindPoints** `spectrum2d_findpoint`（大譜 ~180-480ms/次）：worker 化。
- **Selector** `downsample_points`（O(N²)，5000 點 ~1.3s）：worker 化。
- **線程非進程**：實測 numpy/scipy 釋放 GIL，背景線程不卡主執行緒；避開進程的
  大 signals 陣列 IPC 序列化開銷。
- **OneTone** `find_peaks` 只 0.1ms，**不 worker 化**；最大色散頻率與 inverted slice
  預處理使用 `zcu_tools.analysis.fluxdep` one-tone kernel。它的拖動卡是 redraw 整張 figure，對症優化是
  **重用 scatter(set_offsets) + debounce redraw(50ms)**。

### Flux-Dependence Analysis kernel handoff
ADR-0028 下，互動選點、filtering、line selection、one-tone peak detection 的共用規則住在
`zcu_tools.analysis.fluxdep`。Qt `ui/interactive/` widget 只保留控制項、canvas、worker/debounce
與 Qt event translation；database search、診斷圖與 params export 仍留在 GUI 既有 pipeline。

### flux 對齊：per-spectrum + 可繼承
每張譜各自一份 flux_half/int/period（對齊 persistance.SpectrumResult）。新載入的譜可
`inherit_from` 既有譜的對齊當初值（`alignment_seeded` 標記），LinePicker 才會 seed；
fresh load 用 picker 預設。OneTone 譜的 LinePicker 鎖 magnitude-only（相位無資訊）。

### 跨譜篩選：繼承 min_distance 不繼承 select
Selector 每次開**全選重置**（不繼承 brush 選擇，否則移除的點難加回），但**繼承**
穩定的 downsample threshold(`SelectionState.min_distance`)。

### 配色
互動圖背景一律 `gray_r`（白底、高值=黑），紅點落在高值共振線上對比最強
（vs viridis 高值=黃，紅點不明顯）。對齊 notebook plotly 版的 Greys。

### Remote RPC + MCP（**read-only** observing view，共用 gui/remote transport）
GUI 側：`RemoteControlAdapter` 是第二個 driving-shaped view，但只讀。它 **subclass 共用
`RemoteControlServiceBase`**（`gui/remote/control_service`），後者擁有 router scaffolding
（`route` 骨架 + events.* handlers + `_dispatch_on_main` bare marshal + EventBus
subscribe/serialize/broadcast，底層的 socket/framing/handshake 再委給
`NdjsonRpcEndpoint`）。fluxdep 是 read-only → **零 policy 覆寫**（連 `_get_bus` 都用 base
預設 `ctrl.bus`、event serializers 以 payload `type` 為 key），本檔 `service.py` 只剩
domain 注入（method registry / serializers / 版本 / `server_name="FluxDepRemoteServer"`）。
method 註冊用共用 `MethodSpec`/`build_method_registry`（`gui/remote/method_spec`）。
MCP 側：`zcu_tools/mcp/fluxdep/server.py` 是 thin entrypoint over 共用 `McpBridge`
（`zcu_tools/mcp/core/bridge`：MCP-server-side transport，socket state 是 instance attr）—
config（`fluxdep_` tool 前綴）+ bridge + 3 個生命週期工具。**MCP 層 events dropped**：
MCP bridge 不訂任何 event-push（無 `on_event` hook）；RPC 層的 `RemoteControlAdapter`
則訂閱 7 種 EventBus payload 並透過 `broadcast` 推送給已訂閱的 RPC 客戶端。
- **只讀不變式**：`method_specs`/`dispatch` 只有 6 個純查詢 method
  （`project.info`/`spectrum.list`/`selection.pointcloud`/`fit.result`/
  `resources.versions`/`state.check`），**無任何 mutation**。所有分析（load/align/
  pick/select/fit/export）是 user 在 GUI 裡做；agent 只觀測。原因：選點與軸向判斷需
  人眼看 preview，agent 沒有。`test_dispatch.test_registry_is_read_only` 守這條線。
- **`project.info` / `resources.versions` 用共用 handler**：`dispatch.py` 直接註冊 `gui/remote/readonly_handlers.py` 的 `h_project_info` / `h_resources_versions`（dispersive 用同一份，兩 app 永遠同步）；`_h_state_check` 仍 app-local（用 `gui/project.py` 的 `is_real_project` 判 placeholder）。
- **MCP 工具集**：讀工具自 method_specs 生成（`fluxdep_project_info`/`spectrum_list`/
  `selection_pointcloud`/`fit_result`/`state_check`；`resources.versions` 不曝露）+ 3 個
  生命週期手寫工具（`fluxdep_launch`/`connect`/`disconnect`）。**無 `fluxdep_stop`**——
  agent 不關 user 的 GUI。
- 因 method 全無參數，`method_specs` 的 `params` 為空。
- **省略**（measure-only policy，fluxdep 不需）：version guard / async operation handle /
  diagnostic fan-out / CfgEditor session / render-view——這些都留在 measure 端的
  RemoteControlAdapter + mcp_server，不在共用 transport 裡。

### v2 database search：State 邊界 + 兩條執行路徑
search（`search_in_database`，njit prange 跑數萬筆、釋放 GIL）是 v2 唯一的長阻塞作業。
拆成**純計算 vs State 寫入**兩半，守住 main-thread State 不變式：
- `FitService.compute_search`：純函式，先 snapshot State 的輸入（db 路徑/bounds/transitions/
  選中點雲），再跑 search，**不寫 State**，回 `SearchResult(params, figure)`。可在 worker 跑。
- `FitService.record_result`：唯一寫 State 處（`set_fit_result`），只在主執行緒呼。
- **GUI 路徑（唯一觸發路徑）**：`FitPanelWidget` 的 `_SearchWorker` 跑 compute_search（off-main，
  GIL 釋放不卡 UI），完成 emit `SearchResult` → 主執行緒 slot `record_search_result` 寫 State +
  畫圖。**不可中斷**（單一確定性掃描，只 disable Search 鈕 + 進度條，無 Cancel）。
  - search 是 user 在 GUI 裡按的，**沒有 RPC 觸發路徑**（remote view 只讀）。`Controller.
    search_database` 仍在（GUI worker 用），但不再有 `fit.search` handler。compute/record
    分拆仍是守 main-thread State 不變式的關鍵。
- **進度注入**：`fitting.py` 的 search 走 `make_pbar`（已改，非 gui scope）。GUI worker 用
  `use_pbar_factory` 裝 `GuiProgressBar`（emit Qt signal 到主執行緒進度條，節流 50ms）。

### v2 結果存放 + 視覺化
- `FitState`（State 上的 singleton，version key `fit`）：db 路徑/EJb/ECb/ELb/transitions/r_f/sample_f
  + 結果 params(EJ,EC,EL)。`set_fit_params` 改輸入會清掉舊結果（輸入變則舊結果失效）。
- **AnalyzePanel UI**（`ui/analyze_panel.py`）：selection 後的三步分析集中到**一個「Analyze…」按鈕**
  開的單例面板，內含 **QTabWidget: Filter / Search / Show**（取代舊的 Cross-spectrum filter + Fit
  spectrum 兩按鈕）。
  - **Filter**：嵌 cross-spectrum `SelectorWidget`（進此 tab 時依當前 spectrums 重建）。
  - **Search**：db 搜尋表單（bounds preset 下拉 `general`/`integer`/`all` 填三組 bound spinbox——
    **preset 綁 bounds 不綁 transitions**，transitions 是 `TransitionsForm` 純手填、無自己的 preset）+
    右側診斷圖（QSplitter 可拖拉）。search 前擋空/不存在 db 路徑。
  - **Show**：fit 視覺化 + 顯示工具：x/y 軸上下限數字框（預設按 `viz.derive_auto_limits` = notebook
    `auto_derive_limits`）、r_f/sample_f 參考線 checkbox、要顯示的 transitions 子集（獨立於 fit 用的）。
  AnalyzePanel 是 **MainWindow 持有的單例**（建一次留 stack，切走只隱藏不銷毀），所有 tab 狀態保留。
- **pyplot Gcf 累積坑**：`search_in_database` 的 `plt.figure()` 不 close 會堆進 pyplot 全域 figure 堆疊，
  第二次 search 的 `plt.show()` 會作用在已 detach 的舊 figure → backend raise「not attached」+ 圖只剩標題。
  修法：`_on_search` 每次 `plt.close("all")` 清 Gcf（只丟 pyplot 引用，已內嵌的 canvas 仍活在 container）。
- `transitions` 沿用 `persistance.TransitionDict`（TypedDict + extra_items，混合 r_f/sample_f scalar
  與任意 `transitions{n}`/`mirror{n}` 動態 list 群）——這正是 extra_items 的設計用途，**不改 pydantic/
  dataclass**（會更弱型）。
- `services/viz.py`：matplotlib 重寫 notebook 的 plotly `FreqFluxDependVisualizer`，純函式畫進傳入的
  Figure（background heatmap gray_r + simulation lines + 選中點 + r_f/sample_f const-freq 線 +
  dev_value secondary axis）。診斷圖直接用 `search_in_database(plot=True)` 的後端原生 Figure，不重畫。
- params.json 的 flux_half/int/period 取**第一張已對齊譜**（notebook 單譜語意；多譜同對齊到同 flux 座標）。

## Known Limitations

- **spec_type 持久化**：`dump_spectrums`/`load_spectrums` 現在把 type 存成 h5 group
  attribute（已修；舊檔無 attr 則 type 不設、restore 時 fallback TwoTone）。
- **軸轉置取決於 Labber step channel 順序**：`load_data` 寫死把 `data[:,0,0]` 當 dev、
  `data[0,1,:]` 當 freq，即假設 step channel 是 `[Flux, Frequency]`。但 OneTone 量測常存成
  `[Frequency, Flux]`（freq 掃在外層）→ 軸反。**不是固定特性**（TwoTone 通常正、OneTone 常反），
  要看實際檔案。GUI 的「Transpose axes」toggle（`services/load.py` 的 `transpose_spectrum_data`）
  讓 user 從 preview 判斷後交換。
- **分析全在 GUI（agent 只讀）**：load/定線/選點/篩選/fit/export 都是 user 互動；選點與軸向
  判斷需人眼看 preview。agent 沒有 mutation RPC，只能讀狀態回報，**不可代為操作**。被要求「跑
  分析」時要誠實說明只能 launch GUI + 讀狀態。

## Entry Points

- `script/run_fluxdep_gui.py` — 啟動（`--control-port` 開 read-only RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server；skill `run-fluxdep-gui`
  (`.claude/skills/`，三副本同步 .agent/.codex；`sync_skills.sh` 只同步 SKILL.md) 只含
  SKILL.md。（端到端 smoke.py/make_fixtures 已移除——操作 RPC 沒了，socket 層端到端 smoke
  不再可能；只剩 launch+讀狀態可驗。）
