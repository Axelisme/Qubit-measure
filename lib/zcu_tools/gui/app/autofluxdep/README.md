# `gui/app/autofluxdep/` — Autofluxdep app

**Last updated:** 2026-10-02 — run-owned named figures

這個 app 擁有 autofluxdep GUI shell、workflow 編排、run lifecycle、artifact 與 UI；
[concrete experiment、catalog 與共用量測 mechanics](../../../experiment/v2_gui/autofluxdep/README.md)
由 `experiment/v2_gui/autofluxdep/` 擁有。兩者透過本 app 的 Builder／Node／RunEnv 契約接合，
不把實驗政策放進 Orchestrator。

## Workflow 與執行契約

`nodes/` 定義 Builder、Node、RunEnv、Snapshot、Patch 和 pure-compute predictor。
Builder 宣告 `requires`、`requires_modules` 與產出，並為每個 flux point 建立短命 Node。
Orchestrator 按使用者 placement 次序執行，不做拓撲排序；它解析當點或先前點的依賴，
缺必需輸入時記錄 skip，驗證 Node 的 Patch 後才合併資訊。Builder 的 `make_cfg` 在 Node
取得當點依賴後使用。`cfg/` 擁有 NodeCfgSchema、Generation overrides 和 RunCfgSnapshot；
`feedback/` 擁有通用的 run-local feedback 機制。具體實驗決定各自的 acquire、fit 與 Patch policy。
跨 owner 的 workflow 契約見 [ADR-0062](../../../../../docs/adr/0062-experiment-workflow.md)。

`controller.py` 組合 shared session services，提供 setup、context、device、predictor 與 progress
control facets，並處理 workflow 編輯、run 操作與關閉。`app.py` 建立 core、runtime behavior 與
主視窗。`state.py` 持有 workflow、flux values、run results 和 ProjectInfo。
`ui/` 呈現 node list、cfg form、run progress 與結果圖；plot 建立與更新都在主線程完成。
每次 run 的 Plots 以 node instance name 持有原生 Figure，Builder 接受該 collection 與名稱，
透過 typed factories 建立 subplot。`MainWindow.figures` 提供具名圖的 Mapping；canvas
只負責呈現。Restart／清理／關閉釋放 Qt canvas，不清空已被 caller 保留的 Figure。
`services/remote/` 是 read-only RPC view，不執行 workflow mutation。

## Run lifecycle 與持久化

`run_session.py` 持有單次 run 的 Tools、provider、result、cursor 和 artifact writers。
每個 run／continue segment 是一個 shared OperationRunner operation。Pause 在 flux boundary
暫停並保留同程序 session；Continue 從 `next_flux_idx` 接續。Stop／Abort 是 terminal finalize，
Restart 則建立新 run。Orchestrator 的 node 失敗會回報 run failure，不以空 Patch 代表取消。
本 app 不提供跨 process resume。

RunSession 在 owner thread 固定借用裝置的名稱對應。每個 execution segment 建立新的
RunContext、Plots 與 StopSignal，Node 的 RunEnv 持有同一個 context。Schedule、device
setup 及 provider boundary 共用這個停止來源。Orchestrator 在 Node 返回後讀取同一個
error channel，不讓 partial data 掩蓋失敗。Pause 保留 workflow state，Continue 不沿用
上一 segment 的 token 或錯誤。Run setup 同時固定 State 已觀測的 device snapshot，
Builder 透過 assemble_experiment_cfg 使用它，不在 worker 組裝 cfg 時讀取裝置。Snapshot
與借用的 driver mapping 分開保存。Context 的 plots 是 segment-local non-presenting
collection；UI 的 Result／Plotter 與具名 Figure collection 則存活整個 run，Pause／Continue
沿用同一批 UI 圖。這兩個 owner 不互相接管圖，也不影響 artifact schema。

`services/run_store.py` 和相關 export／report services 擁有 Run Result Artifact。
Metadata root 保存 manifest、journal 和 report；data root 保存 committed node rows 與 exports。
Committed Node Row 與 Flux Point Commit 是不同的 workflow 事實；取消或失敗時已提交的 row
仍保留，未提交的暫存資料不視為 durable。Workflow memento 由 app 的 caretaker 保存；
RunCfgSnapshot 與 artifact 保留本次 run 的配置證據。

Terminal canonical writer、journal 或 manifest 失敗仍使 run 失敗。Terminal sidecar
finalize、export 或 report 失敗不改 finished／stopped outcome，也不清除 sample export
入口；RunStore 返回衍生錯誤，RunSession 與 terminal run payload 的 `output_errors`
保留它們。Qt 另外顯示衍生輸出警告，remote terminal event 投影同一錯誤清單。
Manifest 的 terminal error 和 journal 仍保存所有結算診斷，不等同量測 row failure。
持久化邊界見
[ADR-0063](../../../../../docs/adr/0063-persistence-ownership.md)。

`services/` 也提供 app-local setup、run path、persistence 和 export 的整合；`ui/` 提供
編輯、執行、結果與 read-only inspect 入口。GUI、session 與 operation 的共同分界見
[ADR-0067](../../../../../docs/adr/0067-gui-application.md) 和
[session README](../../session/README.md)。
