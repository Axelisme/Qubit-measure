# Fast-flux transient spectroscopy

此實驗在 flux pulse 的不同時間點掃描 qubit 頻率，用共振位置觀察暫態響應。
它不計算補償濾波器，也不自動寫回校準。

## 開啟與設定

在 measure GUI 開啟 `fastflux/distortion/freq`。
先完成 readout 與 qubit spectroscopy 校準，再確認 qubit alias 與脈衝接線。
不要用模擬器內部參數代替掃描與看圖。

- Reset 取自 `$qubit.module.reset`，缺來源時不使用 reset。
- Flux pulse 使用 `flux_pulse` role，初始為 const waveform、gain 1.0、frequency 0 MHz、Nyquist zone 2，channel 取自 `$qubit.wiring.lo_flux_ch`。本表單設定初始長度 0.5 us。Channel 無法解析時保持 Invalid，不補其他通道。這些是編輯初值，不是校準。執行前必須核對接線、gain、pulse 長度與 pre-delay，或選擇已確認的 library pulse。
- Qubit probe 使用既有 qubit probe 範本，初始 gain 為 0.1、長度為 0.05 us。確認頻率、generator channel 與探測功率適合目前 qubit。
- Readout 有 `$qubit.module.readout` 時保留引用，否則從 readout 範本填寫。
- Length sweep 是 qubit probe 開始前的等待時間，單位 us。初始為 0 到 1 us，共 101 點。
- Frequency sweep 以 `$qubit.freq` 為中心，半寬為 `1.5 * $qubit.kappa`，共 101 點。缺校準值時請手填，並讓頻率範圍涵蓋整條暫態共振。
- `readout_t` 初始為 2 us。它設定平行分支的最短等待時間，不是精確 readout 時刻。
- Reps 與 Rounds 初始各為 100，relax delay 為 1 us。這些是編輯初值，不代表硬體安全設定或充分的熱化時間。

Open 與 Reset 捕捉當時的 alias 和 library。修改全域來源後，按 Reset 才會重新取初值。
Reset 會丟棄目前表單修改。執行前用 preview cfg 核對掃描、pulse duration、pre-delay 與 readout。

## Run 與分析

每次 reset 後，flux pulse、等待再打 qubit probe、以及 `readout_t` 同時開始排程。
所有平行分支結束後才 readout。Flux pulse 的 pre-delay 決定暫態起點。
Length 是 probe 前等待，不是 flux pulse 長度。
Live measurement 圖顯示時間與頻率的 IQ 投影。

按 Analyze 後，核心逐列以 Lorentzian 擬合共振頻率，跳過含 NaN 的列。
圖的時間座標加上半個 probe 長度，表示 probe 中心。
黑點是擬合共振，綠色 Ideal 曲線只是依 flux pulse 時間與資料估計的參考。
灰色區域表示 pulse 邊緣附近的 probe 時間寬度。
請檢查掃描是否跨過共振、邊緣是否被 probe 平均，以及是否有足夠完整資料。
分析需要這次 Run 或 Load 的來源 cfg；缺 cfg 時會拒絕，不使用目前表單補值。
沒有純量校準或 module writeback。

## 保存與離線限制

Save All 沿用核心 native 保存。Signals 的記憶體 shape 為 wait × frequency。
記憶體中的時間是 us，頻率是 MHz；磁碟上的 Wait Time 是 s，Frequency 是 Hz。
Load 只載入結果供分析，不覆寫目前表單。

離線 white-noise 模擬器可驗證排程與 acquisition 接線，但不模擬 flux 對 qubit 頻率的物理響應。
它的圖不能用來判定失真或接受校準。
