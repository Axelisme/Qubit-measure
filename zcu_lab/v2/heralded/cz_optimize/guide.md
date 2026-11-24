**讀者：** operator、agent。用四個實測 herald 類別搜尋雙線 CZ，再用獨立 seeds 驗收。

# Heralded CZ 優化

在 New Tab 選 `heralded/cz_optimize`。每發先做可選的普通 Reset，再用 mux 讀出 herald。每個隨機 Clifford 後播一個 CZ，完成 recovery 後用同一設定讀出終端。只量一個 depth 和 interleaved arm。這個入口不接受只留單一類的 `reset/mux_herald`，也不加 I／Xπ 準備。

## 設定

1. 核對[位置與完整 mux](../../two_qubit/guide.md#綁定位置與讀出-tones)。兩個位置都要有 GE Post 的 circle classifier 與 radius。選已校準的 Xπ/2、Xπ，各 X pulse 與同位置的 CZ pulse 用同一 generator。
2. **Two-line CZ drive** 讀 coupler 的兩個 CZ slots。兩線的起點波形形狀和編譯時長須相同。核對 **Local Z correction pair** 的順序。評估 0 播原 pulse pair 和原 local Z。若用 FAST-DRAG 資產，在 **FAST-DRAG start weights** 填原始三個 I 權重與 β，單位 ns。CMA 直接用這些值起步。用 cosine 時留空，起點只用第一個 I 諧波。候選使用等長 arb，搜尋兩個 **I harmonic ratios**、FAST-DRAG β、共同物理 gain 和兩個 local Z。各 spread 欄位與其他起點的處理方式見[普通 CZ optimize](../../two_qubit/cz_optimize/guide.md#run-前)。
3. **Interleaved Clifford depth** 選在已知 IRB 衰減的中段。填 3–5 個不同的 **Clifford sequence seeds**。每次評估都用這些 seed 身分和相同序列。**Optimizer random seed** 只控制候選。**Avoid frequencies (MHz)** 只在頻譜標位置，不影響分數。

## 分數與停止

每個 seed 只取一輪 **Reps**。對 gg、ge、eg、ee 分別計算終端回到同一類的比例。Qubit1 在前。每類用該 herald 的全部終端計數作分母，終端 Other 算在分母而不算回返。Herald Other 保存但不計分。先等權平均四個類別，再等權平均所有設定的 seeds。固定深度回返率只用來排序，不估衰減參數 `p`、gate fidelity 或 gate error。

任何一個 seed 未量，或四個 herald 類別中有一類的分母為零，該次評估就沒有分數。分母非零而沒有回返是有效的零分。缺類時 Run 完成當次評估的 seeds，再停止。評估 0 缺類時不向 CMA 取候選；CMA 候選缺類時不量同代其餘成員，也不把部分一代交回 CMA。按 **Stop** 不會為了補缺類診斷而繼續收數。Run 保存已完成的 rounds；配置、編譯或取數錯誤照常報錯。

全零包絡不取數，CMA 收到損失 1。它不是實測的零分，不參與勝選，也不觸發缺類停止。

## Analyze 與 Load

- **Selection** 畫出有分數的評估。誤差棒按各 seed／類別的實收發數傳播 shot SEM。有任何一個計分 cell 只收到一發時，仍顯示分數，但不畫它的整體 SEM。誤差棒不代表跨 seed 的散布或 gate error 的信賴區間。
- **Seed returns** 逐 seed／類別畫條件回返率。叉號標出已量的 seed 中沒有 shots 的 herald 類別。無叉號的灰格未量。零包絡同樣未量，但 **Selection** 另標原因。沒有分數時仍可用這張圖查實收資料。
- **Envelopes** 和 **Spectrum** 比較起點與有分數的最佳評估。時間用 µs，包絡幅度用 Qubit1 的 DDS gain 等效尺度。頻譜橫軸相對 CZ 載波。評估 0 的參數列是 CMA 初始均值，不是起點 pulse 的原設定。

**Save All** 用 Native tag `heralded/cz_optimize/fourier-v2`。每個已量 seed 保存 81 格 herald × terminal 聯合計數，各分類軸順序是 g、e、Other。未量格保存 NaN。Native 也保存 seed 身分、三個 I 權重、β（秒）、共同 gain、兩個 local Z、峰值與起點顯示資料。Labber 只匯出具名參數與峰值。**Load** 不讀舊節點版，會指出需要 `fourier-v2`。舊檔請用原寫檔版本開啟。新版用保存的計數重算分數與缺類，不從 NaN 猜測停止原因。若只有 cfg 無法還原，可分析數據，但不能提出寫回。

## 審閱提案與獨立驗收

有效的評估 0 和至少一個有分數的非零包絡候選才能產生提案。缺類停止前已完成的候選也能勝選，即使該代沒有交回 CMA。只有最佳分數嚴格高於評估 0 才預選。沒有改善時提出未預選的原 pulse 與 local Z 設定，不新增 arb 資產。若這些原設定引用的資產缺檔或 SHA-256 改變，則不提出原樣寫回。

CMA 候選勝出時，提案包含一筆共用 arb 資產、兩筆 CZ pulses、兩筆 slots 和一筆有序的 local Z 配對。審閱後再決定是否 Apply。用沒有參與搜尋的 seeds 跑雙臂、多深度的 [heralded RB](../rb/guide.md)，檢查四類共用的衰減、CZ gate error 和信賴區間。不要把搜尋最高分當作改善的獨立證據。

共用操作見[量測流程](../../operator-workflow.md)。
