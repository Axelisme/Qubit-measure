# Dispersive read-only remote

**Last updated:** 2026-10-23

**讀者：** framework-dev、agent。Entry 分析的 RPC/MCP 查詢與版本契約。

`method_specs.py` 是 Qt-free wire 宣告。MCP 從該表產生查詢 tools，
不 import Controller 或 UI。GUI adapter 經 shared RemoteControlServiceBase 處理傳輸，
app-local dispatch 投影 state。WIRE_VERSION=6、GUI_VERSION=8。

`project.info` 回傳 result_root、entry、component、result_path、predictors_path、
dispersive_path 與 predictor。Predictor 是選取的 entry-local JSON 檔名。
未開 entry 時全部為 null。已開 entry、尚未選 predictor 時只有 predictor 為 null。
Identity/path 投影重用 analysis_project.target_payload。

`fit_inputs.info` 回 has_inputs、params、flux_half、flux_bias、flux_period、bare_rf_seed。
params 為 GHz 的 EJ/EC/EL。仿射參數使用 predictor 原生 device units。
未載 inputs 時 has_inputs=false，其餘為 null。
`preprocess.status` 回網格大小與 edelay。
`fit.result` 回 has_result、g、bare_rf、res_dim，頻率為 GHz。
`state.check` 保留 has_project 名稱，表示已選 entry/component target，
其餘 flags 表示 inputs、one-tone、preprocess 與 accepted result 是否存在。
`resources.versions` 仍是內部 bookkeeping 查詢。

MCP 的 dispersive_launch 接受 project_root、entry、component，
另保留 port、token、auto_connect。Project root 預設為 repo root，
entry 可省略讓使用者在 GUI 選擇。Component 沒有 entry 時拒絕。
dispersive_connect/disconnect 保留原連線生命週期。沒有 stop tool。
不新增 load、preprocess、tune、export RPC 或 mutation tools。
