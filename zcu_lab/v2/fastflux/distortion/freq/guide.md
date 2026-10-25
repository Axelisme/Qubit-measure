# Fast-flux transient spectroscopy

**讀者：** operator、agent。從 GUI 或 MCP 在 flux pulse 的不同時間點掃描 qubit 頻率，用共振位置觀察暫態響應。
分析輸出為共振頻率圖，補償濾波器與校準更新須另行處理。

**Last updated:** 2026-10-21

## 開啟與設定

在 measure GUI 開啟 `fastflux/distortion/freq`。
先完成 readout 與 qubit spectroscopy 校準，再確認 qubit alias 與脈衝接線。
以實際掃描與圖形判讀響應，模擬器參數僅供模擬設定。

- Reset 取自 `$qubit.module.reset`，缺來源時停用 reset。
- Flux pulse 模組槽使用 `flux_pulse` 範本，初始為 const waveform、gain 1.0、frequency 0 MHz、Nyquist zone 2。
  Channel 取自 `$qubit.wiring.lo_flux_ch`。本表單設定初始長度 0.5 us。
  Channel 無法解析時保持 Invalid，須填入已確認的通道。這些初值僅供編輯，flux 校準須另行確認。
  執行前必須核對接線、gain、pulse 長度與 pre-delay，或選擇已確認的 library pulse。
- Qubit probe 使用既有 qubit probe 範本，初始 gain 為 0.1、長度為 0.05 us。確認頻率、generator channel 與探測功率適合目前 qubit。
- Readout 有 `$resonator.module.readout` 時保留引用，否則從 readout 範本填寫。
- Length sweep 是 qubit probe 開始前的等待時間，單位 us。初始為 0 到 1 us，共 101 點。
- Frequency sweep 以 `$qubit.freq` 為中心，半寬為 `1.5 * $qubit.kappa`，共 101 點。缺校準值時請手填，並讓頻率範圍涵蓋整條暫態共振。
- `readout_t` 初始為 2 us。它設定平行分支的最短等待時間，實際 readout 時刻由最晚結束的分支決定。
- Reps 與 Rounds 初始各為 100，relax delay 為 1 us。這些初值僅供編輯，硬體安全設定及充分的熱化時間須依實際設備確認。

Open 與 Reset 捕捉當時的 alias 和 library。修改全域來源後，按 Reset 才會重新取初值。
Reset 會丟棄目前表單修改。執行前用 preview cfg 核對掃描、pulse duration、pre-delay 與 readout。

## Run 與分析

每次 reset 後，flux pulse、等待再打 qubit probe、以及 `readout_t` 同時開始排程。
所有平行分支結束後才 readout。Flux pulse 的 pre-delay 決定暫態起點。
Length 設定 probe 前等待，flux pulse 長度由 Flux pulse 模組槽設定。
Live measurement 圖顯示時間與頻率的 IQ 投影。

按 Analyze 後，分析逐列以 Lorentzian 擬合共振頻率，跳過含 NaN 的列。
圖的時間座標加上半個 probe 長度，表示 probe 中心。
黑點是擬合共振，綠色 Ideal 曲線只是依 flux pulse 時間與資料估計的參考。
灰色區域表示 pulse 邊緣附近的 probe 時間寬度。
請檢查掃描是否跨過共振、邊緣是否被 probe 平均，以及是否有足夠完整資料。
分析需要這次 Run 或 Load 的來源 cfg；缺 cfg 時會回報錯誤。
目前表單只供設定後續量測。分析輸出限於共振頻率圖，校準參數及 module 保持原值。

## 保存與離線限制

Save All 保存掃描軸、IQ 與來源 cfg。Signals 的記憶體 shape 為 wait × frequency。
記憶體中的時間是 us，頻率是 MHz；磁碟上的 Wait Time 是 s，Frequency 是 Hz。
Load 載入結果供分析，目前表單保持原值。

離線 white-noise 模擬器的驗證範圍限於排程與取資料流程。
Flux 對 qubit 頻率的物理響應、失真判讀及校準驗收須以實際量測確認。
