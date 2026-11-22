**讀者：** operator、agent。比較兩條 resonator 頻譜，量測 dispersive shift。

## 量什麼
固定 qubit preparation pulse，掃描 readout frequency，比較 preparation 關閉與開啟的兩條 resonator IQ 頻譜。
每個 shot 先執行選用的 Reset，再分別略過或施加 π 脈衝，最後讀出 resonator。
分析以共同 electrical delay 擬合兩條頻譜，回報半個絕對共振頻差 `chi` 與平均 FWHM `avg_fwhm`，單位都是 MHz。
兩條曲線的 ground/excited 標籤只代表 preparation 分支。初態純度與 readout fidelity 須另行驗證。

**Last updated:** 2026-10-21

## 開始前

先完成 resonator 頻率與線寬校準，以及 qubit spectroscopy、Rabi 的 x180 校準。
在 Entry 選 entry、point，確認 `$qubit` 與 `$resonator` aliases 指向這次量測的元件。
確認 readout generator／ADC、qubit generator 與接線相符。
量測期間 flux 保持固定，readout 與 x180 使用已完成的前置校準。

## Open 與 Defaults

1. 在 New Tab 選單選 `twotone` → `dispersive`，開啟實驗。
2. Qubit π preparation pulse 優先使用 `$qubit.module.x180`；沒有時提供可編輯的 pi-pulse template。
3. Readout 優先使用 `$resonator.module.readout`；沒有時提供可編輯的 readout template。
4. Reset 使用 `$qubit.module.reset`，缺少時維持停用。Initial pulse 維持隱藏及停用。
5. Readout RF frequency 初值以 `$resonator.freq` 為中心，完整寬度為三倍 `$resonator.kappa`，共 301 點。確認範圍足以包含兩個共振，必要時擴大。
6. Relax delay 初值為五倍 T1；未存 T1 時使用 30 µs 時間尺度，即 150 µs。已存但無效的 T1 會使相關必填欄位保持空白。

Template 的功率與時序須經硬體安全審核。缺少校準或 channels 時，必須補齊欄位才能 Run。
Open／Defaults 捕捉當下來源。變更 aliases、儲存的 module 或校準後，按 Defaults 才重新取初值；Defaults 會捨棄目前輸入。

## Run 與 Analyze

1. 展開表單，確認 pulse gain、waveform、readout length、Reps 與 Rounds。
2. 按 Run。圖以讀出 RF 頻率 (MHz) 為 x 軸，以平均 complex IQ 的模長 (a.u.) 為 y 軸。藍色是未打 π 脈衝的準備分支，紅色是打 π 脈衝的分支。Reps 是每個準備／頻率組合每輪的 shot 數，Rounds 是平均輪數，初值為 1000 與 10。點數計數顯示有資料的組合。
3. 完成後按 Analyze。`Fit background amplitude slope` 預設關閉；背景明顯傾斜時可啟用後重做分析。
4. 首張 Analyze 圖在兩組量測點上疊加兩條模型與共振垂直線。以兩個共振頻率差的一半取 χ/2π，兩個 FWHM 的平均取 κ/2π。例如中心相差 1.50 MHz 代表 χ/2π = 0.75 MHz。另一張 IQ 診斷圖比較扣除背景並正規化後的兩組 I、Q 點與模型圓。點若有連續偏離圓的區段或共振落在邊界，擴大掃描，或檢查 π 脈衝與背景模型再重測。

Analyze 摘要列該次 Run 的 Reps × Rounds。圖上是平均 IQ 的模長，不是布居。保存結果只有平均 IQ，沒有逐點 shot SEM。數值模型沒有回傳用於這兩條曲線的選取門檻或參數誤差。Analyze 不寫回校準，這個入口目前沒有獨立 recipe。

## 保存與載入

用 Save data 保存本次 Run 的量測設定與兩分支 complex IQ；用 Save artifacts 保存需要的圖與分析產物。
Load data 載入原生格式結果供重新 Analyze，目前表單輸入保持原樣。
保存格式沿用既有 `twotone/ge/dispersive` tag；記憶體頻率為 MHz，磁碟頻率為 Hz，IQ 形狀為 `(2, Nfreq)`。
