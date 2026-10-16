# `zcu_tools.gui.app.fluxdep` flux-dependence analysis GUI

**Last updated:** 2026-10-16

獨立的分析型 GUI，承接 `notebook_md/analysis/fluxdep_fit.md` 的 fluxonium 能譜擬合流程。
領域層獨立於 measure-gui，定位是可選工具，measure-gui 不含它。
Fluxdep 是單一 pipeline session，持有 spectrum 集合，不是平行多 tab；不碰硬體。
共用 transport 與 event 機制位於 `gui/remote` 與 `gui/event_bus`，domain 仍各 app 自帶。

## Pipeline

分析以決策點組織，步驟可反覆執行，順序由人或 agent 決定。
資料依賴仍是對齊 → 採樣 → Filter → fit。下圖列出資料流，不是強制操作順序：

```text
載入 hdf5 spectrum(OneTone/TwoTone)
  → 定線(LinePicker)       → flux_half / flux_int / flux_period(per-spectrum, 可繼承)
  → 選點(OneTone slider / TwoTone brush mask)  → 每張譜的標註點
  → [累積多張 spectrum 進集合]
  → 跨譜篩選(Selector)     → 每張譜的 Filter 遮罩 + min_distance
  → 匯出 spectrums.hdf5
  → [v2] 搜資料庫(FitPanel) → (EJ,EC,EL) + 視覺化 + 診斷圖 → 匯出 params.json
```

Database search 只做 search，不做 scipy fit。`fit_spectrum` 留在 notebook，未移植。

## Owners

分層 `app → Controller → services → State`。MainWindow 是人使用的 driving view；
RemoteControlAdapter 是 agent 的命令與觀測入口。它與 GUI 共用 Controller owners。

| Owner | 職責 |
| --- | --- |
| [`state.py`](state.py) | `FluxDepState` 領域容器，持有 project、spectrum 集合、selection、fit 與版本。集中定義 alignment → points → Filter → fit 依賴圖，另納入 spectrum 集合。 |
| [`controller.py`](controller.py) | 命令 façade，持 State、EventBus 與 service，每動作 mutate State 後 emit 對應事件。無 measure 的 run／writeback／device／tab 概念。 |
| [`services/`](services/) | 薄包裝純運算，mutate State；全部 Qt-free、同步，不碰 bus。純運算核心複用 `analysis.fluxdep` 與 `analysis.spectrum`。 |
| [`interactive.py`](interactive.py) | `FluxDepInteractiveOwner` 每張譜保存一份 live 定線／OneTone／TwoTone context 與 Undo，跨譜 Filter 全域一份。GUI controls 與 RPC commands 共用 `gui.interactive` 的 Actions／Session。 |
| [`search.py`](search.py) | `FluxDepSearchOwner` 集中 single-flight、版本依賴與 terminal policy，提交到專用 `BackgroundRunner`。 |
| [`event_bus.py`](event_bus.py) | Fluxdep payload 型別掛在共用 `BaseEventBus` 上，bus 機制共用、payload 定義 per-app。 |
| [`ui/`](ui/) | MainWindow 的 Spectra／Fit 兩頁、互動 widget 與 AnalyzePanel 單例。切頁只隱藏不銷毀 AnalyzePanel。 |
| [`remote/`](remote/README.md) | `RemoteControlAdapter` 注入 app methods、serializers 與控制身份。MCP entry 位於 `zcu_tools.mcp.fluxdep.server`，GUI 不擁有 MCP tools。 |

## Boundaries and lifetimes

- LoadService 不 import `experiment.v2`，避免把 measure 實驗層拖進來。
  OneTone／TwoTone 載入完全相同，`spec_type` 只是 metadata，下游選點工具才分支。
- 所有 State 寫入只在 Qt 主執行緒。Worker 只計算 detached snapshot，不直接寫 State，
  只經 owner-loop delivery 提供計算結果。
- Active 只決定顯示。切換不退休草稿，同譜換 picker kind 才丟掉該譜草稿。
  TwoTone 初次與重新採樣都從空遮罩開始，不繼承已發布點。
  Source reload、remove 與 external publication 退休對應 context，Filter 仍驗證全部來源版本。
  View detach 不終止 Session。Window close dispose 全部 contexts 並 quiesce runner。
- State 的 search capture 包含所有來源，零點譜也在內。Search delivery、結果過期原因與
  publication invalidated 共用依賴圖。上游發布保留結果並標過期，完整 fit inputs replacement 則清掉結果。
  過期 params 匯出在 I/O 前拒絕。Active spectrum 切換不影響 joint search。
- Cancel 只提出請求，kernel 的 `SearchCancelled` 才代表運算取消。Hide 或 detach 不取消 search。
  關窗先拒絕新 search 並提出 cancel，search runner 未 drain 就拒絕關窗，保留 Qt owners 與圖。
- `Controller.control_identity` 是控制身份的唯一 owner。Shared remote 在 owner loop 檢查 holder；
  讀取只回現況，不建立 per-connection mutation baseline。來源版本與互動 lifetime 仍由 app owner 檢查。
  Fluxdep 沒有硬體緊急 Stop，使用者必須先點燈收回再取消。詳見 [remote README](remote/README.md)。

互動 widget 自持 canvas。Database search 的 [kernel](../../../analysis/fluxdep/README.md)
只算數值，[診斷圖 builder](../../../plotting/fluxdep/README.md) 回傳原生 Agg Figure。
Qt 在主執行緒記錄搜尋結果後，才建圖並透過共用 `gui.plotting` 呈現。
Panel 替換和關窗時 release presentation，舊原生圖仍可保存。診斷圖失敗另報 warning，結果與 export 仍可用。

## Checkpoints and entry points

- Export／Restore 保存採點後的 checkpoint，包含零點完成譜，不保存 Filter 遮罩。
  Params export 保留獨立 sections，取第一張 aligned spectrum 的 calibration，包含零點譜。
  Export 不改 State，失敗保留成功前綴，不 cleanup／rollback。
- 軸轉置取決於 Labber step channel 順序。GUI 的 Transpose axes toggle 讓 user 從 preview 判斷後交換。
  真實資料的工具契約驗證不等於物理校準驗收，校準是否正確仍需另外判讀分析結果與證據。
- `scripts/run_fluxdep_gui.py` 啟動，`--control-port` 開 RPC 給 agent／MCP。
- `.mcp.json` 註冊 `fluxdep-gui` MCP server。Agent 流程入口是
  [run-fluxdep-gui skill](../../../../../.agents/skills/run-fluxdep-gui/SKILL.md)，工具命令與參數由 MCP descriptions 說明；module README 不擁有操作流程。
