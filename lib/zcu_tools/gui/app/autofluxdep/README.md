# `gui/app/autofluxdep/` — Autofluxdep app

**Last updated:** 2026-09-27 — experiment entry relocation

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
`ui/` 呈現 node list、cfg form、run progress 與結果圖；plot 更新在主線程完成。
`services/remote/` 是 read-only RPC view，不執行 workflow mutation。

## Run lifecycle 與持久化

`run_session.py` 持有單次 run 的 Tools、provider、result、cursor 和 artifact writers。
每個 run／continue segment 是一個 shared OperationRunner operation。Pause 在 flux boundary
暫停並保留同程序 session；Continue 從 `next_flux_idx` 接續。Stop／Abort 是 terminal finalize，
Restart 則建立新 run。Orchestrator 的 node 失敗會回報 run failure，不以空 Patch 代表取消。
本 app 不提供跨 process resume。

`services/run_store.py` 和相關 export／report services 擁有 Run Result Artifact。
Metadata root 保存 manifest、journal 和 report；data root 保存 committed node rows 與 exports。
Committed Node Row 與 Flux Point Commit 是不同的 workflow 事實；取消或失敗時已提交的 row
仍保留，未提交的暫存資料不視為 durable。Workflow memento 由 app 的 caretaker 保存；
RunCfgSnapshot 與 artifact 保留本次 run 的配置證據。持久化邊界見
[ADR-0063](../../../../../docs/adr/0063-persistence-ownership.md)。

`services/` 也提供 app-local setup、run path、persistence 和 export 的整合；`ui/` 提供
編輯、執行、結果與 read-only inspect 入口。GUI、session 與 operation 的共同分界見
[ADR-0067](../../../../../docs/adr/0067-gui-application.md) 和
[session README](../../session/README.md)。
