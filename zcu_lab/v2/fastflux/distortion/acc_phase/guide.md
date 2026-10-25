# Fast flux accumulated phase

**讀者：** operator、agent。從 GUI 或 MCP 設定 fast-flux 累積相位掃描並判讀暫態響應。

**Last updated:** 2026-10-21

## 準備

在 measure catalog 開啟 `fastflux/distortion/acc_phase`。
先完成 qubit 頻率、X90 與 readout 校準，再於全域 aliases 綁定
`qubit` 與 `resonator`。量測期間 DC flux 裝置保持原位。

Open 與 Reset 使用固定來源選初值。Reset 會清除本 tab 的編輯並重新捕捉來源。
X90 優先取 `$qubit.module.x90`，readout 優先取
`$resonator.module.readout`。缺少時顯示可編輯範本。
Reset module 為選填，來源是 `$qubit.module.reset`。
Relax delay 初值為已校準 T1 的 5 倍；缺 T1 時使用 150 us，可自行調整。

Flux pulse 模組槽使用 `flux_pulse` 範本，channel 取 `$qubit.wiring.lo_flux_ch`。
缺 channel 時表單保持 Invalid，須填入已確認的通道。範本初值為零頻率、gain 1.0、
Nyquist zone 2 的 const pulse，長度為 1 us。這些欄位都可在表單調整。
初值僅供編輯，flux 校準與硬體安全範圍須另行確認。
執行前須確認 gain、channel、波形、延遲及接線。

## 掃描與序列

Length 是第一個 X90 前的等待時間，單位 us，初值掃描 0 至 2 us、101 點。
Phase 是第二個 X90 的投影相位，單位 deg，初值 0 至 360 度、21 點。
兩個 sweep、reps、rounds 與 readout_t 都可編輯。

每次 reset 後，同時開始三條分支：flux pulse、等待 length 後的第一個 X90、
以及 readout_t 等待。三條都結束後才打第二個 X90，再 readout。
`readout_t` 初值為 2 us，設定平行分支的最短等待時間。
第二個 X90 在最晚結束的分支完成後開始，隨後才 readout。
請將它設得足以覆蓋整個掃描的第一個 X90 與 flux pulse，避免投影時刻隨掃描改變。
Run 前用 cfg preview 檢查實際 pulse、時序與 sweep。

## Run、分析與保存

Run 的 measurement 圖顯示等待時間與投影相位的 IQ 投影。
Analyze 直接使用同一次 Run 或 Load 的 cfg 與 IQ。
它以正弦／餘弦係數抽出並 unwrap 相位，再以負時間斜率除以 2π 得到 MHz detuning。
時間軸對齊第一個 X90 的中心。Fit 圖比較 detuning 與理想矩形響應，
灰色區域標出 X90 在 flux pulse 邊緣附近的有限時間解析度。

請檢查投影對比、相位連續性及邊緣。圖供人工判讀，校正與 fidelity 驗收須另行完成。
梯度與邊緣判讀需要足夠有效點及對比；資料不足時應先改善量測。
分析結果供判讀暫態響應，校準參數及 module 保持原值。

Save All 保存等待時間、投影相位、complex IQ 與該次 cfg。
記憶體等待時間為 us，磁碟為 s，相位維持 deg。
Load 載入結果，目前表單保持原值。歷史 cfg 缺失時仍可載入有效 IQ，
但 Analyze 會回報錯誤，因為它需要來源 pulse 時間。
