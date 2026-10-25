# Dispersive-shift analysis GUI

**Last updated:** 2026-10-25

**讀者：** framework-dev、operator。開啟 entry、選 predictor 與匯出色散擬合的流程及 owner。

## Entry and predictor

以 `python -m apps.dispersive --project-root <root> --entry <entry> --component <name>`
啟動。省略 entry 時，在 GUI 的 Entry 對話框選擇。
省略 component 時，以 setup 的 qubit alias 預選。Alias 缺失或錯誤時，使用者必須明示選擇。

共用 [analysis_project](../analysis_project/README.md) 擁有 entry 開啟、component/kind
驗證、dialog 與 CLI。Dispersive 不複製這些規則。組合根註冊 lab schema，lib 不 import lab。
Project root 下的 result/ 和 Database/ 分別是兩個 roots。分析不執行 Database Save。

Load predictor 列出 entry.predictors_path 下的 JSON 檔名，只接受目錄內的檔案。
`resources.predictor_file` 讀取 EJ/EC/EL 與 flux_half/flux_period/flux_bias。
數值 owner 驗證完整性與物理約束。失敗不替換目前 inputs。
Flux 使用仿射座標 `(device_value + flux_bias - flux_half) / flux_period + 0.5`。
Flux period 可為負數。

Bare resonator frequency 初值依序取：
1. dispersive/<component>.json 的 params.bare_rf。
2. 同檔的 params.plot_transitions.r_f。
3. setup 的 resonator alias 所指 component 已設的 freq，MHz 換 GHz。
4. 5 GHz。

重新載入 predictor 不覆蓋已有 tune bare_rf。Preprocess 完成後，slider 仍以
median_rf 設初值。這兩項分別是 state seed 與畫面初值，不互相取代。
不讀舊 params.json，也不從 chip/qubit 名稱推導目錄。

## Pipeline and export

流程為 predictor → one-tone → preprocess → tune → export。
One-tone 檔可從任意位置選取，預設瀏覽 project_root/Database。
Load dialog 顯示預覽，Transpose axes 決定載入時是否交換軸。
State 永遠保存 canonical device-major signals，頻率由 Hz 轉 GHz。

Tune 的 g slider 顯示 0–200 MHz，r_f slider 的 300 ticks 跨 preprocess 頻率範圍。
Preprocess 後先顯示背景與 bare_rf 線。手動調 slider 或 Auto tune 都不代表接受結果。
Use these g / r_f 執行預測、顯示 ground/excited 色散線並接受結果。
Sample-flux 線支援拖曳，release 後才重算該點。Slider 以 150 ms debounce 重算 sample dots。
Auto tune 以 sample lines 定義 objective，只回填 slider。
固定 PredictionResolution 為 qub_dim=15、qub_cutoff=30、res_dim=4。
數值演算法由 analysis.dispersive 與 simulate.fluxonium 擁有。

Export dispersive 經 `resources.dispersive_file.save_dispersive_file` 寫入
entry.dispersive_path/<component>.json，保存 GHz 的 g、bare_rf 與 UTC created_at。
未知 source.run_id 寫 null。已有 plot_transitions annotations 保留。
Setup 僅更新該 kind 已宣告的 g/bare_rf，不新增欄位。
Artifact 與 setup 是分開的寫入，沒有跨檔原子性或 rollback。
切換 entry/component 會清空舊 pipeline，避免把舊結果匯出到新 target。

## Owners and background work

app.py 組合 State、Controller、MainWindow 與 read-only remote adapter。
State 保存 pipeline 與 VersionTable。Controller 提供命令並發出 domain events。
ProjectService 讀 predictor 與 seed，LoadService 載 one-tone，ExportService 保存結果。
UI 的單一 PipelinePanelWidget 擁有 controls、Preprocess/Tune tabs 與 Matplotlib figures。

Preprocess、predict 與 auto tune 經 BackgroundRunner 在 worker 計算。
Worker 只回傳資料。Owner-thread on_done 記錄 state 並畫圖。
計算中停用 entry/predictor 選擇，避免在途結果跨 target。
關閉視窗前以公開 quiesce_background_work 排空背景工作。

## Remote and MCP

[remote README](remote/README.md) 記錄查詢契約。
Agent 只觀測，GUI 使用者負責 load、preprocess、tune、export。
預設 port 為 8767。Dispersive 的 WIRE=6、GUI=8，與 measure 版本獨立。
MCP 組合根位於 `mcp_servers/dispersive.py`，stdio 實作位於 `zcu_tools.mcp.dispersive.server`。
`dispersive_launch` 使用 project_root、entry、component 啟動組合根注入的 `apps.dispersive` module。
`dispersive_project_info` 回傳 entry/component 路徑與已選的 predictor 檔名，未選時為 null。
MCP 不提供 load、preprocess、tune 或 export 的 mutation tools。
