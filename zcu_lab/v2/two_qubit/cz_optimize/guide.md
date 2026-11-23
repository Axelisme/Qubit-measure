**讀者：** operator、agent。用固定深度的 IRB 尋找 CZ 形狀與 local Z，再以獨立 seeds 驗收。

# CZ 形狀優化

每發做 **可選 Reset → 每個隨機 Clifford 後接一個 CZ → 完整 recovery → 終端 mux 讀出**。同一組 seeds 和序列供所有候選比較；目標是各 seed 的終端 P(g,g) 等權平均，不是 gate fidelity。兩線共用同一 I/Q 形狀；搜尋 I、Q 各六個內部控制點及兩角 local Z，不改各線帶號 gain、載波、相位或延遲。

## Run 前

- 核對[位置與完整 mux](../guide.md#綁定位置與讀出-tones)、各 qubit 校準的 Xπ/2 和 Xπ，以及讀出的 circle classifier 和 GE Post 半徑。各 X 脈衝與同位置 CZ pulse 用同一 generator，phase 為有限值。
- **Two-line CZ drive** 引用 coupler 的 `cz_qubit1`、`cz_qubit2` slots。兩支 pulse 須用同一 arb 資產；兩個路徑不同，即使樣本相同也不收。若要試另一資產，先對兩支 pulse 按 **Convert to inline** 再切換。先核對 I/Q、時長和 normalize，並用該 arb 重測 [CZ amplitude](../cz_amplitude/guide.md) 與 [Local Z](../local_z/guide.md)。**Local Z correction pair** 由 `$coupler.ext.cz_phase` 取有序元件名與兩角，須與本次綁定一致。
- **評估 0** 並非原波形複本。從資產以線性內插取六個等距內部節點，以兩端為零的 natural cubic spline 重建原時長波形，並從已校準的 local Z 開始。起點與原資產形狀可能不同。
- **Interleaved Clifford depth** 選在已量衰減中段；太淺或太深會讓候選分數難以分開。**Clifford sequence seeds** 填 3–5 個不同整數。**Optimizer random seed** 控制候選而非 Clifford 序列。**Initial I/Q control-point spread** 與 **Initial local Z spread** 分別設定 normalized I/Q 和角度（deg）的候選起始標準差。
- 評估數為 `1 + Optimizer generations × Candidates per generation`，嘗試 shots 至多是評估數 × seed 數 × Reps。**Reset** 留白則略過；若用 mux herald，核對[同一讀出與 keep 鍵](../guide.md#herald-reset-與獨立-heralded-入口)。每個 seed 以實際保留 shots 為分母，包含 other；任一 seed 無保留 shots 時 Run 會報該 seed。

## 讀圖與選擇

- **Run** 與 **Analyze fit** 的折線連接實測目標。Run 的虛線為截至各候選的最佳實測值。最大 `|I+iQ|` 超過 1 的候選標記為零，不取數也不進入最佳值。Stop 時未完成候選已保留的 shots 仍存入結果，但不算該候選的目標值。
- **Analyze fit** 比較各已量目標、最大值與起點，沒有 survival 曲線擬合。兩圖的實測點棒是由各 seed 的 retained-shot 二項 variance 傳播到等權平均所得的 shot SEM，不是 seed SD、輪間散布或 fit error。
- **Envelopes** 比較最佳與起點的 normalized I/Q 對包絡時間。摘要依序列 I1… I6、Q1… Q6、Qubit1／Qubit2 local Z。Load 後由保存的時長重建波形，不需原資產；原 cfg 重跑仍需它。缺 cfg 或有效評估就不選最佳。

## 儲存與重載

**Save All** 把候選參數 `params`、波形峰值 `peak`、分類計數 `sums` 和起點時長 `duration` 都寫入 Native `data.h5`。Labber 只匯出 `params` 和 `peak`，兩個 log 各保留自己的軸。要重載完整結果，請開啟 Native 檔。

## 每輪優化與驗收

寫回提案包含一筆新 arb 資產、兩筆改用該資產的 CZ pulse、兩筆 slot 和一筆 `$coupler.ext.cz_phase`。其他 pulse 欄位不變。僅最佳分數高於評估 0 才預選六筆；持平或無法比較不預選。接受時一起採用，讓波形、slots 與 local Z 一致；套用只完成一部分時先核對收據，不用混合設定做 IRB。

用[普通 paired RB／IRB](../rb/guide.md)驗收，確認該表單使用新 pulse 與 phase pair，並換一組未參與優化的獨立 seeds。比較 gate error 和 CI，不用同一組搜尋 seeds 的最高點判斷改善。仍有改善才以已寫回的 12 個控制點和兩角作下輪起點。

舊檔見[家族載入說明](../guide.md#opendefaults-與人工-apply)。共用操作見 [`zcu_lab/v2/operator-workflow.md`](../../operator-workflow.md)。
