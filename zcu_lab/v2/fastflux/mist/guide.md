**讀者：** operator、agent。掃描快磁通與 MIST 驅動強度，找出讀出對比開始變化的位置。

# Fast-flux MIST

每發 shot 先做選用的 Reset 和 Preparation pulse，再平行播放 Flux pulse 與驅動 resonator 的 MIST pulse。兩路都結束才 Readout。結果是平均 IQ，不辨識佔據態或躍遷機率。

## 設定兩路 gain

- 綁定 qubit 與 resonator。確認 Readout 已校準。Reset 與 Preparation pulse 有 qubit 的 reset／x180 slot 時採用，未選則跳過。要比較同一種 preparation，兩次掃描使用相同脈衝。
- 核對 **Flux pulse** 的快磁通接線、pre-delay、包絡長度與允許 gain。核對 **MIST pulse** 的載波、通道和長度。**Flux pulse gain** 和 **MIST pulse gain** 的掃描會覆寫模組的 gain；用已確認的 DAC 振幅範圍。
- **Relax delay** 初值為捕捉的 T1 的五倍；缺 T1 時為 150 µs。

開啟表單、更新來源與保存見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。

## 讀圖與下一步

Live 和 Analyze 的雙 gain 熱圖用全掃描 IQ 最大散布方向的投影（a.u.）。Run 分別顯示有 IQ 的點數和累積輪次。二維圖沒有逐點 shot SEM。舊檔若缺來源 cfg，有效 gain map 仍可分析。

**Probe power** 圖用同一份 IQ，將 MIST gain 改成 gain²；用單一 gain 符號掃描，否則一正一負會落在同一功率值。只有已獨立校準 photon number 對 gain² 時，才填 **Photon calibration coefficient**，圖上估計 photon number = 係數 × gain²。

看到 IQ 位移時，縮小 gain 區間複測。要判斷變化是否依賴 preparation，固定其他設定，比較有無 Preparation pulse 的結果。不從 IQ 顏色直接指認態。本實驗不寫回校準。
