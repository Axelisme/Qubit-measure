**讀者：** operator、agent。從 resonator tone 下的 qubit 頻譜估計平均光子數與驅動增益的關係。

**Last updated:** 2026-10-21

## 設定與量測

先完成 readout 與 qubit spectroscopy 校準。確認 `qubit`、`resonator` aliases 指向正確元件，並檢查 wiring、readout 與 reset。Open／Defaults 從固定來源建立初值。之後若變更 alias，按 Defaults 才重新取初值。Run 使用畫面上的設定。

- Stark resonator tone 使用 `res_probe` 範本，頻率取 `$resonator.freq`，起始長度為 1 µs，與預設 qubit probe 同長。
- Qubit spectroscopy probe 使用 `qub_probe` 範本，起始長度為 1 µs。
- Probe RF frequency 以 `$qubit.freq` 為中心，半寬為 `1.5 * $qubit.kappa`，共 201 點。缺校準時欄位保持未填，請手動設定。
- Stark tone gain (a.u.) 初始範圍為 0 到 0.1，共 21 點。掃描在 gain 平方上等距取點，再做硬體量化。
- Reset 可停用。Readout 優先使用 `$resonator.module.readout`，沒有時提供可編輯範本。
- Reps／Rounds 初值 1000／10，分別是每個頻率每輪的 shot 數與每個 Stark gain 的平均輪數。Relax delay 初值 1 µs，是每個 shot 後的等待。Early-stop SNR 可留空，或讓每條頻譜達到指定 SNR 後提早結束平均。

每個 shot 先執行選用的 reset，再在兩條分支啟動 Stark tone 與 qubit probe。讀出接在 probe 結束後，不會等 Stark tone。調整兩個脈衝的起點與長度，使 probe 量到 Stark tone 下的頻率，避免等到光子已散去才探測。

Run 左圖是 Stark gain 對 probe RF 頻率的熱圖，右圖比較最近兩個 gain 的頻譜。兩圖把平均 IQ 旋轉到主要變化方向，再把每列投影範圍縮到 0–1；色階仍是相對 IQ (a.u.)，不是人口或光子概率。空白區還沒有資料。兩維掃描用來追蹤峰位，不畫逐點 shot SEM。

## 分析與寫回

Analyze 從本次 Run／Load 的固定來源讀取 `$qubit.ext.chi` 與 `$resonator.kappa`，單位為 MHz。缺值時分析回報錯誤；請使用具備必要校準來源的資料。Polynomial degree 可選 1 或 2，對 gain 平方擬合頻率。Gain cutoff 留空時使用全部 gain 資料；設定時排除大於等於該 gain 的資料。

Analyze 在相同熱圖畫出每條頻譜取出的中心和 gain² 多項式。在 Analyze 摘要核對同次資料的 Reps × Rounds。中心追蹤略過不完整列；中心相對前一個接受值的跳動超過 RF 掃描寬度的 10% 時，該列不納入。圖上的 Centers used 是納入數與 cutoff 後總列數，不是擬合優度。若中心跳到另一條線或到達窗口邊緣，減少最高 gain 或移動／加寬 RF 窗口後重測。圖與 summary 的 `ac_coeff` 表示每平方增益的平均光子數。例如係數為 1000 a.u.⁻²，gain 0.03 a.u. 對應 0.9 個平均光子。只在模型跟隨同一條 ridge 時才 Apply 到 `$qubit.ext.ac_stark_coeff`。這個入口目前沒有專用 recipe。

## 保存與載入

Save 保存本次量測設定、gain／frequency 軸與二維 complex IQ。記憶體頻率為 MHz，原生檔案頻率為 Hz。Save／Load 使用原生格式與 tag `twotone/ge/ac_stark`；請選擇相符格式的檔案。Load 保留目前表單輸入；後續 Analyze／Save 只使用載入結果及其來源。
