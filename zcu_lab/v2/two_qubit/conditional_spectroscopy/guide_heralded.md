# Heralded conditional spectroscopy

**讀者：** operator、agent。本頁說明獨立實驗 `two_qubit/conditional_spectroscopy_heralded`。Probe、contrast、DC source、角色捕捉與校準步驟見 [普通版本](guide.md)。

## 序列與校準

每個 shot 執行可選 reset → herald mux readout → 兩顆準備 → 雙線 weak probe → 可選 Xπ⊗Xπ contrast → terminal mux readout。

`prepared_f1`、`prepared_f2` 都是 2 點硬體 loop，0 不發射 pulse，1 發射該 qubit 的 Xπ。F1 在外層。Reset 與 herald 完成後先 sync，再 fork 成兩條 timeline 各做準備，join 後才 probe。兩條 probe 同時播放。Contrast 保留原先的 F1、F2 依序 Xπ。

Herald 與 terminal 重複使用同一個 mux 物件、gain、積分窗口與 circle classifier。Readout 的 post_delay 是 herald 後的等待。F1/F2 tone 都須含 radius，缺少時先做 GE Post。`reset=None` 不發射 reset。Herald 不重設 qubit，也不決定是否執行後續 pulse。

Frequency 在硬體掃描。只有 DC 留在 host，因為需要設定儀器。每個 DC 一支 program 取完兩個準備軸與 frequency，rounds 在取數端累加。

## 依實際初態比較 gg 與 ee

分析與 live 都使用 herald 判為 gg 和 ee 的兩組 terminal 計數。兩組保留各自的 P_e，不合併，也不擬合。Other 留在分母。零選中計數或未量點為 NaN。

每個 DC 各有 F1、F2 頻譜，每張圖的四格是 probe 前的實際準備態。Herald gg 的名目準備 `(f1,f2)` 與 herald ee 的名目準備 `(1-f1,1-f2)` 疊在同格。這裡的翻轉只指 probe 前的準備 Xπ。Probe 後的 contrast 會改變 terminal 分配，圖標題列出是否啟用。

`gg`、`ee` 分析結果仍以名目 `prepared_f1`、`prepared_f2` 儲存。畫圖時才對齊實際初態。兩者的 `acquired_shots` 都是全部 81 類的每點總數，`complete` 以每個頻率是否取得 reps × rounds 判定，不要求每個 shot 都通過 herald。

Shots 圖按名目準備顯示全部 shot 數、選中 gg 與 ee 的數量，以及要求的 shot 數。取消後仍顯示已完成 rounds 的計數，不丟掉部分掃描。

## Herald 九類比例

Herald 圖按 DC 與名目準備顯示各頻率的九類聯合比例。類別依 F1、F2 的 g、e、other 排列。Ge、eg 及含 other 的 shot 都保存，但不進 gg/ee 頻譜。

`pass_fraction` 是 herald gg 計數除以全部 shot 數，**不是 ground-state purity**。它包含 classifier 的誤判與 readout 效應。沒有 shot 的點為 NaN。

## 保存與 DC

Result 保存 float64 `sums`，形狀為 `DC × prepared_f1[2] × prepared_f2[2] × frequency × 3 × 3 × 3 × 3`。末四軸是 herald F1、herald F2、terminal F1、terminal F2，各為 g、e、other。保存全部 81 類總和，不丟棄未選中的類別，也不換成比例。舊 per-shot IQ 與單一 branch 軸格式拒絕載入。

`dc_device=None` 保持單點 DC 哨兵 `[0]`，不動 source。啟用具名 DC 時不補償、不恢復，停止後先確認裝置當下值。Save 保留本次 cfg。Analyze 不從目前表單或 live library 補歷史設定。

## Notebook

先建立同樣通過分類器驗證的 `SpectroscopyHeraldedCfg`，並取得已授權的環境 handles。`point_view` 是保存目的地的 bound PointView。Notebook 經 adapter 管理 scope、plots 與 RunRecord，不直接呼叫 core。

```python
from zcu_tools.notebook import NotebookAdapter
from zcu_lab.v2.two_qubit.conditional_spectroscopy.core_heralded import SpectroscopyHeraldedExp

nb_adapter = NotebookAdapter(soc=soc, soccfg=soccfg, device_manager=device_manager)
exp = nb_adapter(SpectroscopyHeraldedExp())
record = exp.run(cfg)
exp.analyze(None)
saved = exp.save_run(point_view, source=record)
```
