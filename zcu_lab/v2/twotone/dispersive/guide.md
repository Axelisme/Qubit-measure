**讀者：** operator、agent。比較有無 π preparation 的兩條 resonator 頻譜，量 dispersive shift。

## 量測與設定

每個 shot 執行選用的 **Reset** → 略過或施加 **Qubit π preparation pulse** → 掃頻讀出。Ground／excited 是 preparation 標籤，不代表純態。先校準 resonator 頻率與線寬、qubit 頻率與 x180。確認 aliases、readout generator／ADC 和 qubit drive 接線；量測時固定 flux。

- **Readout RF frequency** 以 `$resonator.freq` 為中心，全寬 `3 × $resonator.kappa`；需涵蓋兩個共振及兩側背景。
- **Relax delay** 初值為 `5 × qubit.t1`（未存 T₁ 時 150 µs）。若缺必要校準與時序，補齊後再 Run。
- 背景明顯傾斜時再開啟 **Fit background amplitude slope**。兩條頻譜共用 electrical delay。

共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。

## 讀圖與下一步

Run 與第一張 Analyze 圖的橫軸是 readout RF 頻率（MHz），縱軸是平均 complex IQ 的模長（a.u.），不是布居。藍色不打 π，紅色打 π；Analyze 疊加兩條模型及共振線。此掃描用來定位共振，不畫逐點 shot SEM。

第二張 Analyze 圖以扣除背景且正規化的 I 和 Q 比對量測點與模型圓。若有連續點偏離圓，檢查 π pulse 與背景模型；若共振撞邊，擴大掃描。χ/2π = 兩個共振頻率的絕對差 ÷ 2，例如相差 1.50 MHz 即 0.75 MHz；κ/2π = 兩條 FWHM 的平均。確認 preparation 品質後再使用這些結果。Analyze 不寫回。
