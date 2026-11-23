**讀者：** operator、agent。量測 resonator tone 造成的 qubit 頻移，估計 tone gain 與平均光子數的關係。

## 量測與設定

選用的 Reset → 同時開始 **Stark resonator tone** 與 **Qubit spectroscopy probe** → probe 結束後讀出，不等 Stark tone。先校準 readout 與 qubit 頻譜。核對兩個 aliases 與脈衝時序，讓 probe 取到 Stark tone 下的頻率，而不是光子散去後的頻率。

- **Probe RF frequency** 以 `$qubit.freq` 為中心，半寬 `1.5 × $qubit.kappa`。缺校準時手動填入端點。預期頻移較大時移動或擴大掃描範圍。
- **Stark tone gain** 在 gain² 上等距取點，再量化；不要把相鄰 gain 差當成常數。
- **Polynomial degree** 取 1 或 2，對 gain² 擬合。**Gain cutoff (a.u.)** 留空使用全部 gain；有填時排除大於等於 cutoff 的資料。Analyze 需要該次 Run／Load 來源中的 `$qubit.ext.chi` 與 `$resonator.kappa`（MHz），缺任一值會失敗。

共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。

## 讀圖與下一步

Run 熱圖顯示 Stark gain 對 probe RF 頻率，切線比較最近兩個 gain 的頻譜。平均 IQ 投影至主要變化方向後，每列各自縮至 0–1；色階不是布居或光子概率。空白表示尚未取數。二維定位圖不畫逐點 shot SEM。

Analyze 疊加每列頻譜中心與 gain² 擬合曲線。**Centers used** 比較納入數與 cutoff 後總列數，不是擬合優度。中心追蹤略過不完整列，並排除與前一個接受中心相差超過 RF 掃描全寬 10% 的列。中心若換線或撞邊，降低最高 gain 或移動／加寬 RF 窗口再量。

`ac_coeff` 是每單位 gain² 的平均光子數，例如 1000 a.u.⁻² × (0.03 a.u.)² = 0.9。只有擬合跟隨同一條 ridge 才採用寫回；它更新 `$qubit.ext.ac_stark_coeff`。
