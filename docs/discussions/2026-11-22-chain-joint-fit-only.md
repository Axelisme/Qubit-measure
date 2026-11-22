---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../../lib/zcu_tools/analysis/README.md#readout-chain-k-cycle
  - ../../zcu_lab/v2/README.md#reading-routes
  - 2026-11-22-data-point-lines.md
  - 2026-11-22-fastflux-distortion-fit-failure.md
  - 2026-11-22-ge-gate-split-from-diagnostic.md
basis:
  repo_commit: 7f43d78d6b4098655398c3f07d3dd234bc2e26f7
  review_candidate: 856cb6999092c7ad7558c005d161c3230b1c21e3
  external_sources: none
---

# K-cycle chain 只留主擬合

## 結論

2026-11-22，使用者選 C：「修法的话，我认为就只留第一层，然后 write back 跟分析结果都考虑残差判断就好了。」五個 K-cycle chain 實驗各做一次 joint 主擬合，不再用巢狀擬合檢查主解。使用者也說：「我觉得也不用标模型不符啊…诚实的显示数值就好了。」這次不加寫回。日後若加，殘差診斷用於決定是否預選或推薦，不用於否決 Analyze 的估計。

## 問題

保存的 amp Rabi 掃描在舊版 Analyze 失敗。有些是主擬合耗盡評估次數；有些是主擬合已收斂，但後續每點自由 drive 的 sharing 擬合未過最適性檢查，使 `fit_cycle` 丟棄主結果。只提高主擬合上限沒有解除第二個問題。使用者指出低 SNR 下為診斷增加自由度，反而更難擬合；可以直接看主解的殘差。

## 選項與比較

| 選項 | 做法與取捨 |
| --- | --- |
| A | 保留三層擬合，允許巢狀診斷缺席。可保留主解和成功的附加比較，仍要跑低 SNR 下易失敗的自由擬合，也要定義各層的缺值。 |
| B | 對每點的自由擬合用多個起點，維持原最適性門檻。局部重試救回部分解，但沒有證明完整診斷會跑完，也沒有去掉使用者不需要的兩層。這與替 joint 搜尋不同頻率起點是兩回事。 |
| C | 只做主擬合，檢查同一次解的白化殘差及 live 方向的資料與模型差異。保留判讀線索，不再回答 curve／sharing／within-point 的巢狀比較問題。 |

### 採用 C 後的介面

`CycleCheck`、三種巢狀檢查和由每點自由解畫出的 population 圖一起移除。`fit_cycle` 提高主擬合的評估次數上限，仍只從一個曲線起點擬合。它回傳 `CycleFit`，用最後一階段的模型共變異數白化殘差，給出 `J`、`df`、`J/df`、參考帶與 `|r| > 3` 比例。`cycle_residual_projection` 從已完成的 fit 算出同一 live 方向的實測和預測投影，不再做擬合。投影使用實測 pooled covariance，與計算 `J` 的模型權重不同。各實驗 core 用這些投影呈現點、曲線和帶正負號的差值，並決定哪個估計對應物理量。機器摘要發布標量與缺值，不發布逐點投影陣列。

### 第一讀與投影的失敗邊界

Designer 依授權分別裁定以下兩項，使用者於 2026-11-22 覆核答「可以」。

- **第一讀只供起點。** 保留原本可用的 seed。若第一讀因觀測不足、無對比或已分類的數值失敗而沒有估計，從保存座標取一個起點，用原資料列嘗試主擬合一次。合法短 Ramsey 就算第一個間距為零，也先由 fitting owner 判斷是否缺觀測；足數但不符 FFT 輸入契約仍報錯。主擬合成功時，第一讀圖標「First-read fit unavailable」，連實測點並保留可用的 shot SEM，不畫模型或把起點當估計。原有的第一讀估計欄位報 `null`，不增加新欄位。
- **投影缺席不等於 joint 失敗。** `cycle_residual_projection` 先驗證輸入和所要求的模型網格。只有 Cholesky 失敗且保存矩在算術分辨度內與半正定降秩相容時，才拋出具名的 `CycleProjectionUnavailable`。Core 只隔離這個投影錯，保留原本具名的 joint 圖，標「Live-direction projection unavailable」，列 joint 估計和殘差診斷，不畫假點、曲線或空軸。無效輸入、callback 與主擬合錯誤照常傳播。Live 與 sweep 的既有行為不變。

## 未採用

A、B 都保留了使用者認為不必要的巢狀擬合。第一讀也不改成必須成功，否則只是把丟失主解的條件提前；不一律更換成功的種子，以免改變原有搜尋路徑。投影不加 jitter、不改用 pseudoinverse 或其他方向，因為那會把另一種量當成原 live 投影。

## 留下的問題

單一起點可能收斂到錯的頻率。殘差參考帶和局部 Wald 誤差不能保證物理參數正確。這次沒有加入頻率多起點或寫回，也沒有定義日後寫回的門檻。

## 依據

當時以舊版分析與保存的 amp Rabi 掃描比較三案；原調查工作檔、數據與比較輸出未入庫。固定審查候選又用合成資料分別辨識了「joint 成功卻畫不出投影」和「合法短 Ramsey 被第一讀 FFT 擋住」兩條路徑。實作契約見 [K-cycle 數值模組](../../lib/zcu_tools/analysis/README.md#readout-chain-k-cycle) 與[五族 operator guide 入口](../../zcu_lab/v2/README.md#reading-routes)。

沿用 [GE 寫回保護](2026-11-22-ge-gate-split-from-diagnostic.md) 將數值結果和寫回預選分開的做法。區別 [fast-flux 擬合失敗](2026-11-22-fastflux-distortion-fit-failure.md)：它缺必要的主要 row fit，chain 這兩個分支只缺輔助資訊。第一讀連點和 shot SEM 遵循[量測點呈現規則](2026-11-22-data-point-lines.md)。
