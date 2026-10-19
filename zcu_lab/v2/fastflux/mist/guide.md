# Fast-flux MIST

## 量測內容

`fastflux/mist` 掃描 flux pulse gain 與 MIST pulse gain，觀察平均 IQ 對比。
每個 shot 依序 reset、施加可選的 init pulse，再同時啟動 flux 與 MIST pulse。
兩條分支都結束後才 readout。這張圖不是校準後的態人口或躍遷機率。

## Open 與設定

先完成 readout 與 qubit pulse 校準，再開啟這個實驗。
Open 與 Reset 從固定的 alias capture 讀取 `$qubit` 與 `$resonator`。
Reset 會捨棄目前編輯並重新取初值，改 alias 不會更新已開啟的欄位。

- Reset 與 init pulse 分別使用 `$qubit.module.reset`、`$qubit.module.x180`。
  缺來源時停用，可手動選取 library module 或保持停用。
- Readout 有 slot 就使用 `$qubit.module.readout`，否則使用可編輯的 readout template。
- MIST pulse 使用 resonator probe template。確認頻率、generator 與 waveform 長度。
- Flux pulse 使用共用 `flux_pulse` role，channel 初值來自 `$qubit.wiring.lo_flux_ch`。
  缺 channel 時保持 Invalid，不改用其他 channel。
  Role 的初值為 const waveform、gain 1.0、frequency 0.0 MHz、Nyquist zone 2。
  Role 與此表單的 waveform length 初值均為 1.0 us。
  這些初值不是校準。使用者必須設定 flux gain 範圍，核對頻率、長度與接線。

兩個 gain sweep 的首次範圍都是 0 到 0.1，共 101 點，這不是硬體安全範圍。
Core 會以 sweep 值覆寫對應 pulse 的 gain，改 pulse gain 不會改變 sweep。
請先把兩個 sweep 改成已確認的範圍，再 Run。
Relax delay 以捕捉的 T1 的五倍作初值，缺校準時使用 150 us，可自行修改。
Reps 與 Rounds 分別從 1000 與 100 開始，可先減少掃描點與平均次數檢查訊號。
Run 前核對完整 pulse 與 sweep。

## 看圖與分析

Run 的 `measurement` 顯示 flux gain 對 MIST gain 的旋轉 IQ 對比。
Analyze 的 `fit` 圖將 MIST gain 改成 gain squared。
若已獨立校準 photon number / gain squared，可在分析欄位填入係數，改用 photon-number 軸。
留空時不套 photon 校準。分析只讀本次 Run 或 Load 的資料，不讀目前表單。
這個 adapter 不提供參數或 module writeback。

## 保存與離線限制

Save 與 Load 使用 core 的 native records，保留兩條 gain 軸、complex IQ 與同次 cfg。
Load 不回填目前表單，歷史 cfg 缺失時仍可繪製 gain map。
White-noise backend 可檢查編譯、取資料與 GUI 接線，但不模擬物理 MIST。
FTF 環境只在線性化 DC 附近模擬 flux drive，不支援這裡的大振幅 fast-flux 物理驗證。
