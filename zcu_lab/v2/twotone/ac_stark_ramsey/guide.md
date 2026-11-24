**讀者：** operator、agent。用 Ramsey 條紋的 FFT 比較不同 resonator tone gain 下的 qubit 頻移。

## 序列與設定

先校準 qubit 頻率、π/2 pulse 與 readout。一次量測為 Reset（選用）→ 同時開始固定長度的 **Stark resonator tone** 和 Ramsey 分支（settling delay → π/2 → 自由演化 → π/2）→ 等兩路結束後讀出。確認兩路時序，讓 Stark tone 覆蓋所需的 Ramsey 區段。

- **Stark tone gain** 在 gain² 上等距取點，再量化。延長 **Free evolution time** 的掃描跨度可分辨較小頻差；要分辨 Δf MHz，至少取約 1/Δf µs。最快條紋若為 f MHz，步距至少要小於 1/(2f) µs 以避免 alias，判讀條紋宜每週期取 10 點以上。
- **Phase-ramp detune** 只加在第二個 π/2 脈衝，相位增量為 `360° × detune (MHz) × delay (µs)`。
- **FFT reference frequency** 預設用該次 Run 的 ramp detune，沒有來源時為 0 MHz；修改只影響 Analyze。顯示範圍為 reference ±5 MHz；**Stark gain cutoff** 排除大於等於設定的 gain。

共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。

## 讀圖與下一步

Run 左圖用 gain 與兩個 π/2 間的自由演化時間作軸，色階為主方向的 IQ 投影（a.u.），不是布居。右圖比較最近兩個 gain 的條紋；空白區尚未取數。

Analyze 上圖沿用時間對比圖，下圖以 gain² 對相對 reference 的頻率偏移（MHz）。FFT 模長以各 gain 的最大值正規化，不宜跨 gain 比較強度。二維圖不畫逐點 shot SEM。

以低 gain 為對照，追蹤同一條 ridge。實際峰頻 = reference + 偏移；例如 1 MHz + 0.7 MHz = 1.7 MHz，零偏移不代表無 Stark shift。峰撞 ±5 MHz 邊界時移動 reference 再分析。此入口不擬合 T₂ 或 Stark coefficient，也不寫回。
