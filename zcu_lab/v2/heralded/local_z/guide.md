# Heralded Local Z

**讀者：** operator、agent。量兩顆 qubit 每次 CZ 的 local phase。主估計固定 spectator 為 g，另報 spectator 為 e 時每次 CZ 的相位差。

每發依序做可選 Reset、mux herald、target Xπ/2、n 次雙線 CZ、X 或 Y 投影、mux 終端讀出。Spectator 不打準備脈衝。X 和 Y 使用不同 shots，投影相位相對準備各為 0° 和 90°。Herald 不篩選 shots，也不控制脈衝。每個讀出各有 Qubit1/Qubit2 的 g、e、Other，共保存 81 格計數。

## Run 前

- 綁定 `$qubit1`、`$qubit2`、`$mux` 和 `$coupler`。讀出 tone 位置跟隨 `$mux.qubits`，不是 ADC 編號。兩顆的 tones 都要有已校準的 GE circles 和 GE Post 半徑。
- 核對兩顆的 Xπ/2 與 coupler 的兩個 **Dual-line drive** slots。各 Xπ/2 和對應 CZ drive 用同一 generator。先固定 CZ 振幅、頻率和相對相位，再量 local Z。
- **CZ gates per shot** 填至少三個連續遞增非負整數。預設為 0 到 4；n=0 不發 CZ，可用作基線。
- **Reset** 留空時仍做 herald。要在 herald 前重置時選 pulse reset，不用篩選 shots 的 herald reset。Mux readout 的 post-delay 也用在 herald 後。

## 怎麼讀圖

每顆 target 一張圖。上排是 spectator g，供 local Z 校準。下排是 spectator e，只作對照。左欄畫 X，右欄畫 Y。Herald 標籤依 Qubit1、Qubit2 排列。

- Qubit1 上排用 gg、eg，下排用 ge、ee。Qubit2 上排用 gg、ge，下排用 eg、ee。Target 起始為 e 的組把 X、Y 都反號。每組有自己的 contrast 和相位偏移，同一排的兩組共用斜率。
- 點是對齊後的 `(N_e − N_g) / N_total`，不是純態 Bloch 向量。終端 Other 在分母、對分子貢獻零。Herald Other 留在存檔，不進任何擬合。
- 棒是該組每點的 shot SEM，即樣本標準差除以保留 shot 數的平方根。少於兩發時不畫棒。有資料的點照留。斜率旁的 ± 是擬合標準誤，不是 shot SEM。Reduced χ² 用計數變異衡量曲線與資料的差距，不是接受門檻。
- 少一個必要組或該組無法辨識斜率時，該排只畫已取得的點，標出失敗原因。主估計不改用 gg-only。兩排都有斜率才報 e 減 g，單位為度/CZ，結果折回 [−180°, 180°)。

## 下一步

上排的 correction 等於主斜率反號，單位為度，不再 wrap。用它評估相同 CZ 工作點的 logical frame 修正，不用下排的條件相位代替。此入口不寫回。改動 CZ 脈衝或工作點後重測。一般操作見 [operator workflow](../../operator-workflow.md)。
