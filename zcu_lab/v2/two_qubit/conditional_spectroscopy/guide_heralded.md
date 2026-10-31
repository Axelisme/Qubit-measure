# Heralded conditional spectroscopy

讀者為 operator 與 agent。本頁說明獨立實驗 `two_qubit/conditional_spectroscopy_heralded`。Probe、contrast、DC source、角色捕捉與校準步驟沿用 [普通版本](guide.md)，請先核對該頁的設定。

## 序列與校準

每個 shot 執行可選 reset → herald mux readout → B00/01/10/11 → 雙線 weak probe → 可選 Xπ⊗Xπ → terminal mux readout。

Herald 與 terminal 使用同一個 mux module、gain、積分窗口與 circle classifier。Readout 的 post_delay 就是 herald 後的等待，不另加 delay。F1/F2 tone 均須含 radius，缺少時先做 GE Post。`reset=None` 不發射 reset。Herald 本身不重設 qubit，也不決定是否執行後續 pulse。

## 看圖與 pass fraction

Live 與 terminal 分析只使用 herald 判為 `(g,g)` 的 terminal 計數。兩版共用 P_e 換算，other 留在分母。零通過計數時 P_e 為 NaN，不顯示假分布。圖中的 branch 仍是名目準備，不代表純態。

Herald 圖按 DC、branch 顯示 frequency 對九個 herald 聯合類別的比例。類別依 F1、F2 的 g、e、other 排列。`pass_fraction` 是其中 `(g,g)` 的比例，表示分類接受率，**不是 ground-state purity**。它包含 classifier 的誤判與 readout 效應。

分析的 `terminal.acquired_shots` 是通過 herald 的計數。`terminal.complete` 則以全部 herald 類別的計數判定要求的 reps × rounds 是否完成，不要求每個 shot 都通過 herald。

## 保存與 DC

Result 保存全部 81 類 float64 總和，形狀為 `DC × 4 branches × frequency × 3 × 3 × 3 × 3`。末四軸依序是 herald F1、herald F2、terminal F1、terminal F2，各為 g、e、other。保存時不丟棄 herald 未通過的類別，也不換成比例。舊 per-shot IQ 格式會拒絕載入。

`dc_device=None` 保持單點 DC 哨兵 `[0]`，不動 source。啟用具名 DC 時不補償、不恢復，停止後先確認裝置當下值。取消、取數錯誤與 Load 的規則同普通版本。Save 保留本次 cfg，不從目前表單或 live library 補歷史設定。
