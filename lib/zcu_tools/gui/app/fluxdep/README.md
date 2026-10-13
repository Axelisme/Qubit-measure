**Last updated:** 2026-10-13，GUI 改版（外殼與兩頁、編輯流程、picker 控制欄、Fit 頁、版面收尾）；published spectrum list 與 Filter 排除數

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

## Architecture Overview

分層 `app → Controller(façade) → services → State`。MainWindow 是人使用的 driving
view；RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用
Controller owners。Shared remote 在 owner loop 檢查控制權 holder；讀取只回現況，不建立 per-connection mutation baseline。來源版本與互動 lifetime 仍由 app owner 檢查。
目前支援完整 project／spectrum／selection／fit 查詢、project／spectrum 編輯與載入、
fit inputs replacement、原生 spectrum／params 匯出、app-owned search／operation，
以及四種 shared Session 的互動命令與原生 PNG。

- **`state.py`** — `FluxDepState`（領域容器）：`project`(ProjectInfo)、
  `spectrums: dict[str, SpectrumEntry]`、`active_spectrum`、`selection`(SelectionState)、
  `version`(shared VersionTable)。Spectrum leaf 移除使用精確 retire，刪除期間版本為 0，
  同名重建續增，供來源 replacement 與 context guards 辨識；不影響同字首的其他譜。Collection／global keys
  保持 State lifetime 的計數。
  **`ProjectInfo`/`default_*` 共用** `gui/project.py`（Qt-free，與 dispersive 同源）；
  `ProjectDialog` 共用 `gui/widgets/project_dialog.py`（`db_label="Data root"`），並可從
  project root 掃描到的 `result/**/params.json` result scope 下拉選取既有 chip/qubit。
  `SpectrumEntry` 持 raw(SpectrumData)/points(PointsData)/per-spectrum flux 對齊/
  aligned/points_completed/alignment_seeded/filter_mask。Filter 遮罩依點 index 儲存，
  set_points、reset_points 與同名取代只清掉該譜的遮罩；對齊保留遮罩。
  selection_mask() 依插入順序投影成聯合點雲的 bool 陣列，未篩選的譜全選。raw/points 直接複用
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
- **`interactive.py`** — Qt-free `FluxDepInteractiveOwner` 每張譜保存一份 live 定線／OneTone／TwoTone context 與 Undo，跨譜 Filter 全域一份。GUI controls 與 RPC commands 共用 `gui.interactive` 的 Actions／Session。
  Active 只決定顯示。切換不退休草稿；同譜換 picker kind 才丟掉該譜草稿，open receipt 回 `discarded`。
  Source reload、remove 與 external publication 退休對應 context，Filter 仍驗證全部來源版本。
  每份 line plugin 有獨立 single-flight auto alignment；共用 background runner 可並行提交不同譜。
  Sampling intent 依已發布譜型開 OneTone 或 TwoTone picker。TwoTone 初次與重新採樣都從空遮罩開始，不繼承已發布點。
  Finish 經 Controller services 發布。Window close dispose 全部 contexts 並 quiesce runner；view detach 不終止 Session。
- **`event_bus.py`** — fluxdep 的 payload 型別，掛在共用 `BaseEventBus`
  （`gui/event_bus`，payload-type-key 訂閱）上；bus 機制共用、payload 定義 per-app。
- **`ui/`** — `MainWindow`（上方列 + Spectra／Fit 兩頁，見「主視窗外殼」）、互動 widget
  （`ui/interactive/`）、`load_dialog`/`export_dialog`。共用層（與 dispersive）：
  `LoadSpectrumDialog` subclass `gui/widgets/load_dialog.LoadDataDialog`（共用 file
  row/transpose/preview/OK-gate，`_build_options` 掛 Type/Inherit combo、`result_request`
  回 `LoadRequest`）；`error_messages` 用共用 `gui/error_messages` framework（domain rule
  各 app）；`ui/paths.nearest_existing` 來自 `gui/project`；背景正規化（`contrast_limits`、
  `onetone_dip_depth`）在 `plotting.fluxdep.image`，供 result_preview 與 OneTone picker 共用。MainWindow 擁有的 EventBus subscriptions 在 window close
  釋放，避免分析 view 被 bus callback 保活。
- **`remote/`** — `RemoteControlAdapter` subclass 共用 `RemoteControlServiceBase`
  （`gui/remote/control_service`），注入 app methods、serializers 與控制身份。MCP
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

### 主視窗外殼
上方列左邊是 project 標籤（`Project: <chip> / <qubit>`，chip 與 qubit 都是預設值時顯示提示）與控制權燈，
右邊是 Project…、Add spectrum…、Restore…、Export spectra…。主區是 Spectra 與 Fit 兩頁 tab：
Spectra 頁是譜清單（Remove 在清單下方）加階段驅動編輯區，Fit 頁承載 AnalyzePanel 單例。
選譜、active 切換與 stage 變更只重建 Spectra 頁的編輯區，不蓋掉 Fit 頁。切頁只跟人類意圖：
使用者點 tab，或 GUI Add／Restore 後回到 Spectra。Agent-origin 的 active、picker、Filter 與
search pending 只更新所屬頁的內容，不把使用者從目前的頁拉走（見「Interactive context facts 與 GUI 跟隨」）。
離開 Fit 頁只 detach Filter view，進入 Fit 頁經 `show_tab` 重新 attach。
清單每列取 State 與 interactive owner 的同一份事實（即 `spectrum.list` 的資料）：字面譜名、種類、
階段（loaded／aligned／sampled）、已發布點數，有才加未完成草稿種類與 Filter 排除點數。
測試以 objectName 定位外殼元件（`fluxdep.session.project`、`fluxdep.pages`、`fluxdep.page.*`、
`fluxdep.spectra`、`fluxdep.editor`、`fluxdep.project`、`fluxdep.spectrum.*`、`fluxdep.spectra.*`）。

### 編輯區階段驅動
MainWindow 編輯區顯示 active 譜的 live 草稿；沒有草稿時，本機 holder 的人類意圖（點清單、GUI Add）
經 holder gate 開目前階段的 picker：未定線→LinePicker；已定線未完成選點→OneTone/FindPoints(by spec_type)。
其餘情況顯示 ResultPreview（已發布狀態的唯讀圖）。GUI Add 成功後直接開新譜的定線 picker；
RPC 載入與 active 切換只更新顯示，不開 picker。
空 Finish 也完成選點，清單顯示完成，ResultPreview 顯示零點。
`points_completed` 表示階段完成；`point_count` 表示可用點數。Analyze／Selector 只使用非空資料。
widget 的 `finished` → Controller 寫回 → 階段前進 → 重 swap。
ResultPreview 只列當下可用的下一步：未對齊只有 Align；已對齊有 Re-align 與 Sample。
三者經同一 holder gate 開該階段的 picker 草稿，已發布的對齊與點保留到 Finish 才換掉；
TwoTone 重新採樣照舊從空遮罩開始。Sample picker 有 Re-align，重新定線的 LinePicker 有 Sample，
讓使用者在兩階段間切換。同譜已有另一階段的未 finish 草稿時，GUI 先用非阻塞的確認對話框詢問，
取消則草稿與畫面不變；確認後 owner 照舊丟掉舊草稿（RPC 的 `discarded` 投影不變）。
Processed restore 包含零點的完成結果，不新增持久化欄位。

### Picker 控制欄
定線、OneTone 與 TwoTone 三個 picker 共用 `InteractiveMplWidget` 版面：圖在左、220px 控制欄在右。
控制欄由上而下是設定、編輯動作（Undo／Clear）、狀態文字；階段切換與 Finish 固定在欄底。
每個 slider 與下拉都有 buddy label。控制項 objectName 為 `fluxdep.picker.<action>`
（例如 `finish`、`undo`、`clear`、`auto_align`、`status`、`realign`、`sample`），
ResultPreview 也是 `InteractiveMplWidget`：圖在左，下一步按鈕固定在右欄底部，objectName 為
`fluxdep.preview.<align|realign|sample>`；測試以這些名稱定位，不依按鈕文字。
定線資訊用固定小數位，位數由 device 軸步長決定；mirror loss 圖有 Device value／Frequency 軸標。
canvas 每次 resize 重新 tight layout，軸標不會被窄視窗裁掉。

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
`inherited_from` 表示目前已發布對齊的來源。未改繼承線直接 Finish 保留來源；
手動改線或 Auto Align 後 Finish 才清除。丟棄草稿不改已發布對齊或來源，
Controller 直接發布手動對齊則清除來源。

### Interactive context facts 與 GUI 跟隨
Owner 的 inspect 回傳 live context reference 與 owner-lifetime identity，不建立 Session，也不消耗 Undo。
同一有效 begin 重用 identity；退休後的新 context 使用新 identity。Identity 不代表 edit revision，
同一 Session 的 GUI／command Actions 仍依 owner-loop 順序提交。

Owner 發布 opened／updated／closed domain facts，Session commit／undo 與定線 alignment info 都更新同一 context。
MainWindow 依 EventMeta origin 與目前 identity 在 Spectra 頁編輯區顯示 agent 開啟／修改的 picker，
active 跟著切到該譜，但不切頁。Agent 的 Filter 與 search pending 只更新 Fit 頁：Fit 頁正顯示時切到
Filter／Search tab，否則記住該 tab，使用者下次進 Fit 頁時落在那裡。
GUI 依 context kind 掛載，不能用已完成的 pipeline 階段代替目前 context。
Closed 只清除旧 view，不開新 Session。純讀取與 user-origin edits 不強制切換畫面。
AnalyzePanel 的 public show_tab 管理 Filter／Search／Show，外部不存取 Qt 私有 stack 或 tab。
Search terminal 更新結果，不搶回使用者已切換的畫面。

### 跨譜篩選：繼承 min_distance 不繼承 select
App owner 建立新的跨譜 context 時全選，僅繼承已發布的 `SelectionState.min_distance`。
Analyze singleton 每次進入 Fit 頁的 Filter 都重新 attach 有效 context；離開 Filter tab 取消 context，切到 Spectra 頁則只 detach view。
Selector controls／完整 stroke 透過共用 Actions 修改 Session。Preview 使用 80ms debounce 與 generation guard；
Apply 不依賴 preview worker，teardown 阻擋 hidden 舊 controls 與晚到 delivery。

### 配色
互動圖背景一律 `gray_r`（白底、高值=黑），紅點落在高值共振線上對比最強
（vs viridis 高值=黃，紅點不明顯）。對齊 notebook plotly 版的 Greys。定線圖與 mirror loss 圖也用
`gray_r`；它們的 renderer 是共用的 `analysis.fluxdep.TwoLinePicker`，所以 notebook 的 InteractiveLines
與 MCP 的定線 PNG 同樣是 `gray_r`。OneTone 採點 picker 與 ResultPreview 的 OneTone 背景用同一種正規化：
平滑後的 magnitude 取 dip 深度（最大值減 magnitude），再以 2–98 百分位定色階，共振 dip 是白底上的深色線。
採點 picker 的 renderer 是共用的 `plotting.fluxdep.onetone.OneTonePickPlot`，所以 MCP 的 OneTone 採點 PNG 也跟著改變。

### Remote RPC 與 MCP

`RemoteControlAdapter` 注入 app methods、event serializers 與 application control identity。
共用 remote 擁有 route、owner marshal 與 holder admission，endpoint 擁有 socket、
framing、authentication 與回覆交付。Read 不建立 seen baseline，寫入預設要求 holder；
readonly method 才允許 observer。詳見 [remote README](remote/README.md) 與 ADR-0070。

Project、spectrum、selection 與 fit reads 提供目前 owner projections。
Spectrum snapshot 包含 calibration、published points 與 raw axes extents，不輸出 complex
signal matrix。Selection snapshot 提供 published mask 與 normalized min_distance，
不讀 live Session。Remove、reset 與 active selection 共用 Controller owners。
Raw load 保留 basename replacement、inheritance／transpose 契約。
Processed restore 是採點後的 checkpoint，包含零點完成譜。同名衝突在發布前拒絕整批，列出衝突名稱。
Export 保存所有已採點，不保存 Filter 遮罩。其他 publication 失敗保留已發布前綴，不 rollback。
Fit replacement 使用 `search_db`，與專案 raw-data root 區分。JSON parser 驗 type／shape／finite；kernel 在發布前驗 transition 類別與能階索引，只有需要能階數才讀 DB metadata，不載入 energies。拒絕或 I/O 失敗不改舊 inputs／result，成功則清掉舊結果，不啟動搜尋。Bounds 與缺少頻率仍在 search 時驗證。
GUI Show 與 `fit.render` 共用 Controller 的 native overlay 入口。顯示範圍、reference lines 與 transition 子集只改 presentation，不改 search inputs。`fit.database_info` 只讀 DB counts 與參數範圍。

Spectrum export 預設 create-only，明示 overwrite 才替換。Params export 沿原生 merge，
保留 sections，取第一張 aligned spectrum 的 calibration，包含零點譜。
`FitService.export_params_receipt` 在一次 owner-thread 匯出中捕獲 immutable `FitExportReceipt`，
包含成功路徑、全精度 GHz 的 EJ／EC／EL 與實際校準譜名 `alignment_from`。
Controller 提供同名入口，RPC `fit.export_params` 投影該收據，MCP 不重讀狀態或推導來源。
既有 `export_params` 取收據的路徑並維持 str 回傳，Qt 與 Notebook caller 不變。
沒有 fit、fit 過期或沒有已對齊譜時，在 I/O 前拒絕；I/O 失敗不回成功收據。
Export 不改 State；失敗保留成功前綴，不 cleanup／rollback。
Search 使用同一 app owner token，status 可找 GUI 或 agent 的最新 activity。
Await 在 IO thread 等待，不碰 State；timeout 不取消，cancel receipt 不代表 cancelled outcome。
Unknown／evicted token 明示拒絕，結果由 fit.result 讀取。

Interactive open 以譜名定址，不要求 active。Read 與 command 按 name 取得同一份草稿。
Read 的 optional context_id 與 command identity 保留 owner-lifetime guard。
數值回覆不帶 commands 或 figure；readonly `interactive.render(name)` 單獨產生 Agg PNG。
Command 核對 identity 與 target 後呼叫同一 plugin Actions、Session Undo 與 app publication。
TwoTone mask-cell count 與 detected-point count 分開；selection 保留 input mask 與
downsample 後 kept count。Finish／cancel receipt 保留退休 identity，即使 GUI 已開下一份
context。Filter finish 發布並關閉全域 context；Apply 仍不關閉有效 joint input。
readonly selection.render 固定畫已發布點雲與各譜遮罩，interactive.render 只畫 live draft。
兩者都不建立 Session。publication／rendering 失敗不宣稱 rollback。
顯示切換不開 Session。Local holder 的人類點譜可另開目前階段 picker，observer RPC 只顯示草稿或已發布資料。
PNG 不使用 Qt canvas，也不保存為產物。

MCP entrypoint 使用共用 bridge，手寫工具組裝基礎 RPC 並摘要資料，control tools 轉送同一
authority。MCP 不訂閱業務 event-push，不維護另一套 holder 或 operation registry。
RPC clients 可訂閱 EventBus facts。Agent 不關閉使用者的 GUI。

### Qt 控制身份與編輯停用

`Controller.control_identity` 是控制身份的唯一 owner。共用 `ControlIndicator` 顯示
Local、Local 鎖定及 Remote label／在線狀態。點燈呼叫 `toggle_local`，收回成 Local
鎖定或切換 Local 鎖定；轉移本身不取消 search 或 interactive Session。

MainWindow 的 `set_writable` 停用 collection／project／IO 命令與
redo controls，spectrum 清單與 Spectra／Fit 切頁仍可用。主視窗推送狀態給 Analyze panel 和目前 picker。新 view 套用當下狀態，
子元件不訂閱 authority。Picker 停用 controls 與 domain pointer gestures，
canvas 保持可用，接管時丟棄未提交的 Local pointer preview。
Local modal 期間 owner 取得回 `local_busy` 與「Local 使用中」。Qt remote adapter
在 owner turn 查 active modal，chooser 使用非 native option。Dialog 不接 control，
chooser 返回與 accepted callback 不重查 holder。

Analyze 的 Search form（含 Cancel）在 Remote 停用；Search 與 Show 的狀態列在表單外，非 holder 仍可讀。Filter tab 的進出會開啟或
取消 Session，所以在 Remote 停用；無 Filter input 時 Search／Show 切換及 Show
的 display controls 保持可用。非 holder 可進 Fit 頁閱讀結果與 `Outdated: …`；沒有 live Filter
context 時進入 Search tab，不為閱讀開 Filter input，有 live context（例如 agent 的）則 reattach。
Filter view detach 仍只是 teardown，不收回控制權。
Fluxdep 沒有硬體緊急 Stop，使用者必須先點燈收回再取消。

Remote 關窗先確認，拒絕不開始 shutdown。接受後沿原 search cancel／drain 與
interactive dispose 契約處理，不自動收回控制權。Close 釋放主視窗的 authority watch。背景 drain、app-owned Session 與 EventBus
subscriptions 保留原 lifetime。

### Database search 的 app ownership
`FitService.capture_search` 在 State owner capture detached `SearchInput`。
`compute_search(inputs)` 只算數值，不讀 live State、不繪圖。
`record_result(result, inputs)` 在 owner 寫入同次 `SearchInput` 的原始 normalized-flux／GHz 點雲、點序與版本，Controller 發布 `FitChanged`。Kernel 的 modulo-one flux 不取代背景上的原始座標。
State 集中定義 alignment → points → Filter → fit 依賴圖，另納入 spectrum 集合。Search capture 包含所有來源，零點譜也在內。Search delivery、結果過期原因與 publication invalidated 共用此圖。
Alignment／points／fit-input stage counters 留在 GUI 內部，不出現在 resources.versions；Filter 與集合重用既有 counters。`spectrum:<name>` 仍供 context guards，`fit` 仍記結果 availability。結果發布不改 fit-input counter。
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

### Caller-correctable errors

State、SearchOwner 與 load／fit／export owners 使用 shared `InvalidInputError` 與
`FailedPreconditionError` 分類 caller 可修正的失敗。State 的 `get_spectrum` 是 owner-thread
literal-name lookup，回傳 live entry；未知名稱不修改來源或版本。Remote 只投影 nominal category
與 reason，不從 ordinary exception ancestry 或訊息猜分類。Missing runtime、thread misuse、
provider I/O 與 worker failure 保留各自的 unexpected failure／operation outcome 語意。

### v2 結果存放 + 視覺化
- `FitState`（State 上的 singleton，version key `fit`）：db 路徑/EJb/ECb/ELb/transitions/r_f/sample_f
  + 結果 params(EJ,EC,EL)、kernel quality／attribution、原始 search cloud 與捕獲版本。`set_fit_params` 改輸入會清掉舊結果。
  品質保留完整精度 GHz、編譯順序與重複 labels，無 recorded quality 時為 null。
  四種 picker finish 回 `PublicationReceipt`，包含 accepted 數值與 Controller 在 emit 前捕獲的 invalidated。同步事件 subscriber 開 successor 或再發布都不能改掉本次 receipt。Filter Apply 維持數值回傳。
- **AnalyzePanel UI**（`ui/analyze_panel.py`）：selection 後的三步分析集中在**主視窗 Fit 頁**承載的
  單例面板，內含 **QTabWidget: Filter / Search / Show**。
  - **Filter**：嵌 cross-spectrum `SelectorWidget`（進此 tab 時依當前 spectrums 重建）。
  - **Search**：搜尋表單（`Search DB` 是 fit 的 `search_db`，與專案的 Data root 分開；bounds preset 下拉
    `general`/`integer`/`all` 填三組 bound spinbox，下拉永遠顯示與 spinbox 數值完全相符的 preset，數值被改過時顯示 `custom`——
    **preset 綁 bounds 不綁 transitions**，transitions 是 `TransitionsForm` 純手填、無自己的 preset，各列以畫出的頻率命名）+
    右側診斷圖（QSplitter 可拖拉，圖佔剩餘寬度；控制欄靠上，表單不拉滿高度）。輸入的 transitions 需要而 r_f／sample_f
    留空時，欄位下方直接標示；Search 不在面板預檢頻率，由 Controller 拒絕。search 前擋空/不存在 db 路徑。
    `Browse…` 用非阻塞 `open()` 的非 native chooser，選檔後填入 `Search DB`；開著時 owner 取得回 `local_busy`。
    Search 的錯誤（含 search 失敗）寫在 Search 狀態列 `fluxdep.fit.status`，不跳 modal；params export 的長路徑只作
    status/tooltip，不參與 panel 水平最小寬度。按鈕 `Search`、`Cancel`、`Export params…`。
  - **Show**：fit 視覺化 + 顯示工具：x/y 軸上下限數字框（Auto limits 取 `Controller.fit_auto_limits`，演算法同 notebook
    `auto_derive_limits`；已記錄 search 用同次 capture，即使過期，搜尋前與 params-only 用已發布來源）、r_f/sample_f 參考線 checkbox、要顯示的 transitions 子集（獨立於 fit 用的）。
    繪圖問題（子集缺頻率、無法繪製）寫在 Show 自己的狀態列 `fluxdep.show.status`，不寫進 Search。
    Show canvas 每次 resize 重新 tight layout，縮小視窗時標題與軸標留在 canvas 內。
  - 測試以 objectName 定位 Fit 頁元件：`fluxdep.fit.*`（search_db、bounds_preset、r_f／sample_f 與其 `.required`、
    search、cancel、export_params、status、freshness）與 `fluxdep.show.*`（apply、auto_limits、status）。
  AnalyzePanel 是 **MainWindow 持有的單例**（第一次進 Fit 頁時建立，切頁只隱藏不銷毀），所有 tab 狀態保留。
  Panel 訂閱已發布的 fit 變更，只比較已發布的 search inputs：inputs 改變才重載 Search 與 Show 表單，結果、品質或捕獲版本改變時保留本地 display edits。
  Panel 也訂閱 spectrum 集合、spectrum 與 selection 事件：Search 的 Freshness 列與 Show 標題寫出 State 的 stale reasons，過期時停用 Export。
  Show 的 display choices 分 applied 與 pending：Apply 取走全部欄位並重畫；自動刷新只用 applied choices，只在結果版本或 `pipeline_snapshot` 的來源版本改變時重畫（同一原因文字的第二次對齊也重畫，但已記錄 search 的背景維持 capture，fresh search 後才換新背景），不清掉、也不套用尚未 Apply 的欄位。新結果只套用 auto limits 的數值。
  Visualization 的 fallback 使用已發布的 fit transitions，不使用未提交的 Search 表單。
- Search 診斷图 builder 建立原生 Agg Figure，不登記 pyplot manager。Panel 替換或關閉時 release presentation，不以全域 `plt.close("all")` 管理其他 caller 的圖。
- `transitions` 沿用 `analysis.fluxdep.models.TransitionDict`（TypedDict + extra_items，混合 r_f/sample_f scalar
  與任意 `transitions{n}`/`mirror{n}` 動態 list 群）——這正是 extra_items 的設計用途，**不改 pydantic/
  dataclass**（會更弱型）。
- `services/viz.py`：matplotlib 重寫 notebook 的 plotly `FreqFluxDependVisualizer`，純函式畫進傳入的
  Figure（background heatmap gray_r + simulation lines + 選中點 + r_f/sample_f const-freq 線 +
  dev_value secondary axis）。`Controller.render_fit` 畫 recorded search cloud，點依 compiled transition 上色，偏大殘差點空心，標題列 State 的 stale reasons。`Controller.render_selection` 畫 published masks，並疊 stored fit 曲線，不標 residual，也不讀 live draft。兩者共用 kernel 能階需求與 simulation，合法高階 transition 不讀 DB 也不漏線。診斷圖直接用共用 builder 的後端原生 Figure，不重畫。
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
- **物理校準驗收**：真實資料的工具契約驗證不等於物理校準驗收。校準是否正確仍需另外判讀分析結果與證據。

## Entry Points

- `scripts/run_fluxdep_gui.py` — 啟動（`--control-port` 開 RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server。Agent 流程入口是 `.agents/skills/run-fluxdep-gui/SKILL.md`，工具命令與參數由 MCP descriptions 說明；module README 不擁有操作流程。
