# `zcu_tools.gui.app.fluxdep` flux-dependence analysis GUI

**Last updated:** 2026-10-19

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

**只做 search，不做 scipy fit**（fit_spectrum 留在 notebook，未移植）。

## Architecture and owners

分層 `app → Controller(façade) → services → State`。MainWindow 是人使用的 driving
view；RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用
Controller owners。

- [`app.py`](app.py) 只做 controller/window/adapter wiring；程序入口位於 `scripts/run_fluxdep_gui.py`。
- [`state.py`](state.py)： `FluxDepState`（領域容器）。
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
TwoTone preview completion 只更新 presentation，不提交 Session 或 spectrum points。

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
RPC clients 可訂閱 EventBus facts。Agent 不關閉使用者的 GUI。

`Controller.control_identity` 是控制身份的唯一 owner。
Fluxdep 沒有硬體緊急 Stop，使用者必須先點燈收回再取消。

### Database search 的 app ownership

`Controller.search` 的 `FluxDepSearchOwner` 集中 single-flight、版本依賴與 terminal policy。
State 集中定義 alignment → points → Filter → fit 依賴圖，另納入 spectrum 集合。
上游發布保留結果並標過期，`set_fit_params(search_db, bounds, transitions, r_f, sample_f)` 的完整 replacement 則清掉結果。`bounds` 重用 analysis owner 的 `ParamBounds(EJ, EC, EL)`，三組上下限以 GHz 表示。Controller、FitService 與 State 轉交同一組 bounds，不改 RPC 欄位或 UI 控制項。過期 params 匯出在 I/O 前拒絕。
Hide 或 detach 不取消，重新 activate 讀 owner snapshot。
關窗先拒絕新 search 並提出 cancel，search runner 未 drain 就拒絕關窗，保留 Qt owners 與圖。

[Search owner](search.py) · [Fit service](services/fit.py) · [依賴圖與版本](state.py)

### Checkpoints and export

Processed restore 是採點後的 checkpoint，包含零點完成譜。
Export 保存所有已採點，不保存 Filter 遮罩。
Export 不改 State；失敗保留成功前綴，不 cleanup／rollback。

[Load service](services/load.py) · [Export service](services/export.py) · [匯出收據](services/fit.py)

## Known limitations

- **軸轉置取決於 Labber step channel 順序**：要看實際檔案。GUI 的「Transpose axes」toggle 讓 user 從 preview 判斷後交換。
- **物理校準驗收**：真實資料的工具契約驗證不等於物理校準驗收。校準是否正確仍需另外判讀分析結果與證據。

## Entry points

- `scripts/run_fluxdep_gui.py`，啟動（`--control-port` 開 RPC 給 agent/MCP）。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server。Agent 流程入口是 [run-fluxdep-gui skill](../../../../../.agents/skills/run-fluxdep-gui/SKILL.md)，工具命令與參數由 MCP descriptions 說明；module README 不擁有操作流程。
