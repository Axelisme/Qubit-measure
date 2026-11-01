# Readout optimization SNR dtype

**讀者：** operator、lab-author、framework-dev、agent。辨識 readout frequency／power 優化的實數 SNR 與既有 native 檔的讀回方式。

## Result 與存檔宣告

`twotone/ro_optimize/freq` 與 `twotone/ro_optimize/power` 的 run 產生 float64 SNR。
原本 `AxesSpec.Z` 未指定 dtype，讀回時使用預設的 complex128。
現在兩個實驗明寫 `dtype=np.float64`，保存與讀回都使用實數 SNR。
這項修正不改 SNR 數值、序列、掃描順序或分析方法。

Frequency 軸仍在記憶體使用 MHz、磁碟使用 Hz，比例為 `1e6`。
Power 軸的 label 與 unit 保持不變。
Signal 的 label 仍為 `Signal`，unit 仍為 `a.u.`。
Native container 與 cfg schema 版本不變。

## 既有資料

以舊宣告保存的 SNR 可以使用 complex128，但虛部為零。
現行 native loader 可直接把這些值讀回 float64，不需要轉換工具或重寫檔案。
驗收使用修正前 writer 保存的 frequency／power native 檔，確認修正後 loader 成功讀取，且座標與 SNR 值不變。
非零虛部不是這兩個實驗的 SNR，loader 不會為了讀取它而丟棄虛部。
