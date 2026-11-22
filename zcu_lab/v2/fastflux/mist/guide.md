**讀者：** operator、agent。掃描 fast-flux 脈衝與 MIST resonator drive 的 gain，判讀讀出對比何時出現變化。

# Fast-flux MIST

開啟 `fastflux/mist`。每個 shot 先執行選填的 Reset 及 Preparation pulse，再同時安排 Flux pulse 和 MIST pulse。兩路都結束才 Readout。MIST pulse 在 flux 偏置期間驅動 resonator，本實驗觀察其後的平均 IQ 對比，不直接辨識哪個態被佔據或測得躍遷機率。

## 設定

Open 或 Defaults 取當時選定 point 的 qubit、resonator 和 library。更換來源後按 Defaults，會捨棄目前編輯。Reset、Preparation pulse 分別取 qubit 的 reset 與 x180 slot，缺少或停用就不播放。Readout 優先用 resonator 的讀出 slot；缺少時填入校準設定。

Flux pulse 用 qubit fast-flux 通道。範本初始為 1 µs 矩形包絡、0 MHz、Nyquist zone 2。先核對接線、gain 範圍及預延遲。MIST pulse 用 resonator probe 範本，核對載波、通道與長度。Flux pulse gain 與 MIST pulse gain 分別掃兩路的 DAC gain，覆寫模組內的 gain。兩個掃描的初值都是 0 到 0.1 a.u.，各 101 點；請選已確認的範圍。

Reps 預設 1000 shots／點／輪。Rounds 預設取整張圖 10 輪。Relax delay 預設為捕捉的 T1 的五倍，缺 T1 時是 150 µs。若想讓準備保持相同，固定其餘脈衝設定後再比較兩張圖。

## 看圖

Run 和 Analyze 第一張圖用兩路 gain 作座標。色階是平均 complex IQ 沿全掃描最大散布方向的投影，單位 a.u.。沒有逐發分類，不能把顏色當態機率。Run 輪進度顯示累積平均；圖的標題顯示有 IQ 的點數。二維掃描不畫逐點 shot SEM。

Probe power 圖保留同一份 IQ，把 MIST pulse gain 換成平方，便於沿功率看分支或起始變化。已獨立校準 photon number 對 gain² 時，於 Analyze 填 Photon calibration coefficient，縱軸便顯示「估計 photons = 係數 × gain²」。不要從這張圖猜係數。做功率對照時使用單一 gain 符號，因為正負 gain 的平方相同。

若在固定 flux gain 下出現突然的 IQ 位移，先縮小 gain 區間重掃。例如變化落在 0.04 到 0.06 a.u.，可用 0.035 到 0.065 a.u.、31 點，把步距縮到 0.001 a.u.。改變 Preparation pulse 或降低 MIST gain 複測，區分可重現的對比和背景。不從 IQ 顏色寫回態校準。

## 保存與離線限制

Save 保留兩條 gain 軸、complex IQ 和來源 cfg。Load 不改目前表單；即使舊 cfg 不可用，仍可畫有效的 gain map。沒有 recipe 或寫回。White-noise 驗證只涵蓋離線取數和作圖，不模擬 MIST。

FTF 的 `DriveLine.operator: flux` 是 DC 工作點附近的小訊號算符。`flux_lines` 是另一條路由，可對序列內的 fast-flux 波形施加時變磁通模型。本實驗沒有驗證 `flux_lines` 對大振幅脈衝或 MIST 的物理適用性；判讀須靠實機複測。
