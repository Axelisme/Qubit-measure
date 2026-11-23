**讀者：** operator、agent。以 flux pulse 前後各時間點的 qubit 共振頻率，辨識快磁通暫態。

# Fast-flux transient spectroscopy

每發 shot 先做選用的 Reset，再同時開始 Flux pulse、等待 → 掃頻 qubit probe，以及 Readout t。等最晚分支完成才讀出；Readout t 只設定最短等待。

## 設定掃描

- 先完成 readout 與 qubit spectroscopy 校準。核對 qubit、resonator、探測功率與接線。Flux pulse 通道初值取自 `$qubit.wiring.lo_flux_ch`；缺值會留 Invalid。選已確認的通道與 gain，並以 pre-delay、包絡長度設定 flux 的前後緣。
- **Length** 是 probe 前的等待時間（µs）。掃描涵蓋脈衝前基線、脈衝及尾端；probe 太長會平均邊緣附近的響應。
- **Frequency** 是絕對 MHz，初始為 qubit 頻率 ± 1.5 倍 qubit 線寬。若缺校準值請手填；讓整條暫態共振都落在頻窗內。確認 Readout 的積分時序與每發後的 Relax delay。

開啟表單、更新來源與保存見 [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md)。

## 讀圖與下一步

Live 以時間與頻率顯示平均 IQ 投影的熱圖，不是布居，也沒有逐點 shot SEM。Analyze 的橫軸是 probe 中心（Length + 半個 probe 長度）。黑點是每列 Lorentzian 擬合的中心頻率；灰帶是以程式指定的 flux 邊緣為中心、各一個 probe 長度的時間窗。

綠色 **Ideal** 矩形以程式指定的邊緣和資料估得的高低準位作比較，不是量到的理想波形。要估基線至少要有 10 個成功擬合的時間點。Analyze 跳過含 NaN 的列；若一列擬合報錯，這次 Analyze 不出新圖，已採資料與 Live 圖仍可檢查。沒有本次 Run／Load 的來源 cfg 時無法分析 pulse 時序。

先檢查共振是否被頻窗截斷，以及是否需要更短的 probe 或更長的時間窗。這個分析不估補償濾波器，也不寫回校準。
