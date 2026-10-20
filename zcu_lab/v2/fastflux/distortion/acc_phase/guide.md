# Fast flux accumulated phase

## 準備

在 measure catalog 開啟 `fastflux/distortion/acc_phase`。
先完成 qubit 頻率、X90 與 readout 校準，再於全域 aliases 綁定
`qubit` 與 `resonator`。本實驗不移動 DC flux 裝置。

Open 與 Reset 使用固定來源選初值。Reset 會清除本 tab 的編輯並重新捕捉來源。
X90 優先取 `$qubit.module.x90`，readout 優先取
`$qubit.module.readout`。缺少時顯示可編輯範本。
Reset module 為選填，來源是 `$qubit.module.reset`。
Relax delay 初值為已校準 T1 的 5 倍；沒有 T1 時使用 150 us，可自行調整。

Flux pulse 使用共用 `flux_pulse` role，channel 取 `$qubit.wiring.lo_flux_ch`。
缺 channel 時表單保持 Invalid，不猜通道。Role 初值為零頻率、gain 1.0、Nyquist zone 2
的 const pulse，長度為 1 us。這些欄位都可在表單調整。
這不是 flux 校準。執行前須確認 gain、channel、波形、延遲及接線。
不要把此範本當成硬體安全保證。

## 掃描與序列

Length 是第一個 X90 前的等待時間，單位 us，初值掃描 0 至 2 us、101 點。
Phase 是第二個 X90 的投影相位，單位 deg，初值 0 至 360 度、21 點。
兩個 sweep、reps、rounds 與 readout_t 都可編輯。

每次 reset 後，同時開始三條分支：flux pulse、等待 length 後的第一個 X90、
以及 readout_t 等待。三條都結束後才打第二個 X90，再 readout。
`readout_t` 初值為 2 us，是 join 的最低時長，不是保證的精確 readout 時刻。
請將它設得足以覆蓋整個掃描的第一個 X90 與 flux pulse，避免投影時刻隨掃描改變。
Run 前用 cfg preview 檢查實際 pulse、時序與 sweep。

## Run、分析與保存

Run 的 measurement 圖顯示等待時間與投影相位的 IQ 投影。
Analyze 使用同一次 Run 或 Load 的 cfg 與 IQ，沒有額外分析選項。
它以正弦／餘弦係數抽出並 unwrap 相位，再以負時間斜率除以 2π 得到 MHz detuning。
時間軸對齊第一個 X90 的中心。Fit 圖比較 detuning 與理想矩形響應，
灰色區域標出 X90 在 flux pulse 邊緣附近的有限時間解析度。

請檢查投影對比、相位連續性及邊緣，不把圖當成自動校正或 fidelity 驗收。
沒有足夠有效點或對比時，梯度與邊緣判讀不可靠。本 adapter 不提出 writeback。

Save All 使用核心原有保存格式，保存等待時間、投影相位、complex IQ 與該次 cfg。
記憶體等待時間為 us，磁碟為 s，相位維持 deg。
Load 不改目前表單。歷史 cfg 缺失時仍可載入有效 IQ，
但 Analyze 會明確拒絕，因為它需要來源 pulse 時間。
