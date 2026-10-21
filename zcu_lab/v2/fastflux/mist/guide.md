# Fast-flux MIST

**讀者：** operator、agent。從 GUI 或 MCP 設定 fast-flux MIST 掃描，並判讀兩條 gain 軸上的 IQ 對比。

**Last updated:** 2026-10-21

## 量測內容

`fastflux/mist` 掃描 flux pulse gain 與 MIST pulse gain，觀察平均 IQ 對比。
每個 shot 依序 reset、施加可選的 init pulse，再同時啟動 flux 與 MIST pulse。
兩條分支都結束後才 readout。圖中的 IQ 對比須經額外校準才能解讀為態人口或躍遷機率。

## Open 與設定

先完成 readout 與 qubit pulse 校準，再開啟這個實驗。
Open 與 Reset 讀取當時的 `$qubit` 與 `$resonator` alias。
修改 alias 後，按 Reset 重新取初值。Reset 會捨棄目前編輯。

- Reset 與 init pulse 分別使用 `$qubit.module.reset`、`$qubit.module.x180`。
  缺來源時停用，可手動選取 library module 或保持停用。
- Readout 有來源就使用 `$qubit.module.readout`，否則使用可編輯的 readout template。
- MIST pulse 使用 resonator probe template。確認頻率、generator 與 waveform 長度。
- Flux pulse 模組槽使用 `flux_pulse` 範本，channel 初值來自 `$qubit.wiring.lo_flux_ch`。
  缺 channel 時保持 Invalid，須填入已確認的通道。
  範本初值為 const waveform、gain 1.0、frequency 0.0 MHz、Nyquist zone 2。
  Waveform length 初值為 1.0 us。
  這些初值僅供編輯，flux 校準須另行確認。使用者必須設定 flux gain 範圍，核對頻率、長度與接線。

兩個 gain sweep 的首次範圍都是 0 到 0.1，共 101 點，硬體安全範圍須另行確認。
量測以 sweep 值覆寫對應 pulse 的 gain，掃描範圍由兩個 gain sweep 欄位決定。
請先把兩個 sweep 改成已確認的範圍，再 Run。
Relax delay 以捕捉的 T1 的五倍作初值，缺校準時使用 150 us，可自行修改。
Reps 與 Rounds 分別從 1000 與 100 開始，可先減少掃描點與平均次數檢查訊號。
Run 前核對完整 pulse 與 sweep。

## 看圖與分析

Run 的 `measurement` 顯示 flux gain 對 MIST gain 的旋轉 IQ 對比。
Analyze 的 `fit` 圖將 MIST gain 改成 gain squared。
若已獨立校準 photon number / gain squared，可在分析欄位填入係數，改用 photon-number 軸。
留空時使用 gain squared 軸。分析只使用本次 Run 或 Load 的資料。
分析輸出限於圖形，校準參數及 module 保持原值。

## 保存與離線限制

Save 與 Load 保留兩條 gain 軸、complex IQ 與同次 cfg。
Load 載入結果，目前表單保持原值。歷史 cfg 缺失時仍可繪製 gain map。
White-noise 模擬器的驗證範圍限於編譯、取資料與 GUI 接線。
FTF 環境的 flux drive 模擬限於線性化 DC 附近。
物理 MIST 與這裡的大振幅 fast-flux 響應須以實際量測驗證。
