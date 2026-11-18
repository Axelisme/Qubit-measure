---
date: 2026-11-18
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-14-sampled-multitone-interim.md
basis:
  - "repo main@1ae2449966675122bc7a6b896247b691e248e7dd"
  - "repo candidate@5de43357e53440aab2e87a817a16664b37f8bc6b"
  - "ABBA baseline@9a09218c725426f435da63a2707c9a4add9bf84d"
  - "ABBA and RB candidate@28afb820f26198c0910f1c5cb35e92c7f2ceb74e"
---

# Sampled multitone RWA 的實測放行

## 結論

這次恢復 sampled 雙 tone 列的 `multitone_rwa` 資格，不恢復 `multitone_gate`。候選先過逐列指標，再把只差各 tone 振幅倍率的同形列分成一群。每群取各 tone 振幅最大的列並去重。用同一固定探針，在每個抽中的列比較 RWA 與完整 drive `dop853` 的密度矩陣。

估價把抽查算進各群自己的成本。預設選路先確認該群有機會比合格的替代路徑便宜，才抽查。每列抽查差不超過 `0.1 × budget` 才讓整群走 RWA；任何抽查失敗，整群交回原候選。分群、數值與選路的現行規則由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 說明。

## 問題

[過渡期拒收](2026-11-14-sampled-multitone-interim.md)是為了防止局部數值指標替完整序列的 density 背書。代價是 `simultaneous_drive` 這類 sub-GHz 雙 tone sampled 列都走完整 drive `dop853`。其中 24 點、同時打兩個成形 X90 的增益掃描是這次要加速的工作負載。固定探針必須便宜到能攤回整群，又不能讓異形列借到不適用的抽查結果。

## 考慮過的選項與比較

- **以完整脈衝或切割列分群。** O2 的 23 個重疊列在切割邊界的包絡都不為零，這個分法沒有隔開形狀與長度不同的列。按振幅族分群時，24 點增益掃描形成一群，O2 的 23 列各成一群。單列群不值得為了 RWA 再跑一次完整 drive。
- **保留固定探針。** O2 診斷的一列用實際入口態得到的差約為固定探針的 2.5 倍。這小於 `0.1 × budget` 到 `budget` 的 10 倍餘量，但不是其他場景的誤差上界。
- **給拒收案例硬性 1.1 倍上限，或記錄並在超過 1.2 倍時回報。** 採後者。本例中抽查沒過才退回原 family，因此快場景可能多付一筆固定成本。這是本次驗收的回報線，不是數值放行門檻。

正式成對計時用 CPU、7 workers、三個 ABBA block。對已含 CPU DP5(4) 的基準，24 點、d24 的 `simultaneous_drive_retained24` 約從 956 ms 降到 304 ms，中位加速 3.149 倍。候選對獨立參考解的完整 density 最大差為 `5.855e-6`。d4 拒收場景耗時是基準的 1.168 倍，兩側完整 density 相同。另一次診斷量到抽查約 232 ms，其中約 230 ms 是該列已編譯的完整 drive 參考積分。診斷用來拆分成本，不拿來取代 ABBA 計時。

使用者於 2026-11-18 對採用建議的方案答「嗯，都采用建议」。同日對拒收回報線說：「以比较快的场景稍微变慢一点为代价，让慢场景快很多是可以接受的，所以可以采用你的建议。」

使用者同日把納入與倍率的判斷交給 designer：「你们要帮我决定啊，因为你们最清楚各项参数的细节。你应该知道这个算法在什么情况下有优势，什么情况下有劣势，值不值得的，要不要纳入，要不要采取变种，是你们帮我决定的。」DP5(4) 讓基準快約 1.6 倍，抽查成本主要花在完整 drive 積分。Designer 因此將舊基準的 5 倍門檻改成新基準的 3 倍，接受這次結果。

## 被否決的選項與理由

- **另抽低邊界振幅的列。** O2 誤差最大列的邊界振幅只有 2.5%，這個見證在 O2 碰巧有效。不同形狀的列沒有共同的誤差上界，故不把這個樣本當通用見證，也不按場景名稱或長度特判。
- **加多個探針，或改用實際入口態。** 多跑完整 drive 會縮小加速。用實際入口態還會要求查詢先演化使用者的 timeline。這次用固定探針搭配獨立完整 density 驗收，不把查詢改成模擬執行。

## 留下的問題

d32、depth 50 的 RB 序列在預設選路下，完整 density 誤差外推至 depth 500 為 `4.958e-3`，低於 `5e-3`。65 個 sampled pair 都走 `dop853`，沒有 RWA offer。既有 capture 不足以歸因，因此這筆結果驗收預設精度，不量 RWA 在 RB 的加速。完整 pinned `dop853` 對照在 600 秒內未完成。

在掃描中，抽查的完整 drive 積分仍是最大的新增固定成本。本次沒有找到不改探針或精度契約就能移掉這筆成本的改法。

## 依據

現行模組邊界見 [ADR-0078](../adr/0078-evolution-module-boundary.md)。[Accuracy matrix](../../tests/simulate/evolution/contract/test_accuracy_matrix.py)保留門檻內側、整群拒收與獨立完整 density 的合成案例。正式 ABBA 與 RB 報告的候選是 `28afb820f`，不是後來的審查修正樹；原始工作檔未入庫。

Land review 修正了 GPU chain 比價、共享前綴列座標與 common-frame guard 係數交接。用 CPU `inspect_dispatch` 比對修正前後的正式兩景、O2、O1 與三個 C2b，217 列的 family 均相同。這次核對只驗選路，沒有重跑 ABBA、RB 或凍結參考解。
