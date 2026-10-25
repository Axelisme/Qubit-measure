# Design analysis

**Last updated:** 2026-10-21

**讀者：** lab-author。模型參數搜尋、設計候選與實測樣本比較工具。

`add_real_sample(..., samples=entry.records(component), device_name=...)` 要求 caller 明示 `JsonlRecordTable`，也接受 caller 自行綁定的表。`result_dir` 只提供 `params.json`，不推論樣本路徑或元件。函式使用該表的 `read()`，從 `devices` 投影指定裝置後依 resolved flux 選 row，再取 `t1_us` 繪製實測值。JSONL reader 每個 run_id 只回傳最後一行，缺少 flux 的 row 需要呼叫端提供同單位的 fallback frame。HFSS sweep 仍讀 CSV。

這裡評估模型參數、設計需求與候選組合，不建構版圖。`search.py` 建立 `(EJ, EC, EL)` 參數表，計算能譜、頻率、矩陣元、T1 與讀出 SNR，以碰撞及門檻條件篩選候選，並在 SNR–T1 圖上標出比較結果。`hfss.py` 讀取 HFSS sweep CSV，分析與參考頻率的距離及 anticrossing。`snr.py` 計算 photon sweep 上的讀出 SNR。用法見 [`notebook_md/analysis/design.md`](../../../../../notebook_md/analysis/design.md)。

本目錄組合評估指標；物理模型由 `zcu_tools.simulate` 擁有，設計搜尋呼叫其中的 fluxonium / Floquet 計算。此流程與 [`notebook/circuit_design`](../../circuit_design/README.md) 的電路建構和 GDS 輸出彼此獨立，沒有固定上下游；它也不與 GUI fluxdep search 合併。

待核實：範例 Notebook 在特定設備資料與模擬環境中的端到端執行結果；此 README 僅描述已見於模組程式與 Notebook 的用途。
