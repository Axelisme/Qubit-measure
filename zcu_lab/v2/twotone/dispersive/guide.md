# Dispersive shift

## 量什麼

**讀者：** operator、agent。在 GUI 或 MCP 量測 dispersive shift。
固定 qubit preparation pulse，掃描 readout frequency，比較 preparation 關閉與開啟的兩條 resonator IQ 頻譜。
每次序列先 Reset，再選擇略過或施加 qubit pulse，最後 readout。
分析以共同 electrical delay 擬合兩條頻譜，回報半個絕對共振頻差 `chi` 與平均 FWHM `avg_fwhm`，單位都是 MHz。
兩條曲線的 ground/excited 標籤只代表 preparation 分支。初態純度與 readout fidelity 須另行驗證。

**Last updated:** 2026-10-21

## 開始前

先完成 resonator 頻率與線寬校準，以及 qubit spectroscopy、Rabi 的 x180 校準。
在 Setup 選 entry、point，確認 `$qubit` 與 `$resonator` aliases 指向這次量測的元件。
確認 readout generator／ADC、qubit generator 與接線相符。
量測期間 flux 保持固定，readout 與 x180 使用已完成的前置校準。

## Open 與 Reset

1. 在 New Tab 選單選 `twotone` → `dispersive`，開啟實驗。
2. Qubit preparation pulse 優先使用 `$qubit.module.x180`；沒有時提供可編輯的 pi-pulse template。
3. Readout 優先使用 `$qubit.module.readout`；沒有時提供可編輯的 readout template。
4. Reset 使用 `$qubit.module.reset`，缺少時維持停用。Initial pulse 維持隱藏及停用。
5. Readout freq 初值以 `$resonator.freq` 為中心，完整寬度為三倍 `$resonator.kappa`，共 301 點。確認範圍足以包含兩個共振，必要時擴大。
6. Relax delay 初值為五倍 T1；未存 T1 時使用 30 µs 時間尺度，即 150 µs。已存但無效的 T1 會使相關必填欄位保持空白。

Template 的功率與時序須經硬體安全審核。缺少校準或 channels 時，必須補齊欄位才能 Run。
Open／Reset 捕捉當下來源。變更 aliases、儲存的 module 或校準後，按 Reset 才重新取初值；Reset 會捨棄目前輸入。

## Run 與 Analyze

1. 展開表單，確認 pulse gain、waveform、readout length、Reps 與 Rounds。
2. 按 Run。量測圖顯示兩條 frequency 對 amplitude 曲線；Reps 與 Rounds 初值為 1000 與 100，可依雜訊及時間調整。
3. 完成後按 Analyze。`Fit background amplitude slope` 預設關閉；背景明顯傾斜時可啟用後重做分析。
4. 檢查兩條頻譜的 fit 與 normalized IQ circles。若共振不在掃描範圍、prepared branch 沒有可信差異或擬合不佳，先檢查範圍、初始化與 x180 校準，確認資料可信後才採用 `chi`。

分析呈現本次資料的數值與圖，校準保持原樣。

## 保存與載入

用 Save data 保存本次 Run 的量測設定與兩分支 complex IQ；用 Save artifacts 保存需要的圖與分析產物。
Load data 載入原生格式結果供重新 Analyze，目前表單輸入保持原樣。
保存格式沿用既有 `twotone/ge/dispersive` tag；記憶體頻率為 MHz，磁碟頻率為 Hz，IQ 形狀為 `(2, Nfreq)`。
