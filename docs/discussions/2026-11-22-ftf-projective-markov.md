---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0074-flat-readout-module.md
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - ../../lib/zcu_tools/simulate/adc/README.md
  - 2026-11-17-projection-markov-chain-readout.md
  - 2026-11-21-projective-span-reuse.md
  - 2026-11-22-simulator-speed-first-verification.md
basis:
  investigation_commit: ac971b13eea6f5cb004a486cdd24856aab63612a
  measured_candidate_commit: e602f0dc6d11997744bd41b11580012baf37d20e
  repo_commit: 07b53828254ed151c2e012ef73a7cd618f2bed7d
  external_sources: none
---

# FTF projective 從真空續演化並改用 Markov chain

## 結論

使用者選擇分兩步改 FTF。第一步在每次 DC 投影後把入射腔場設為真空。第二步讓同點、同層、同 DC 態共用一列續演化，直接交給 ADC 的 `ProjectionChain`，不再按 outcome history 算一列。逐點 floor 對低機率來源態使用 identity，不丟質量。這沿用 single／six 的 [Markov 先例](2026-11-17-projection-markov-chain-readout.md)，也取代原先 FTF 要保留場記憶與逐 history 演化的邊界。使用者 2026-11-21 對兩步方案答「採用建議」。現行規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#projective-read-的-markov-chain-表示)。

Compact 直取契約另由 designer 代裁，使用者未覆核。ADC 沿同一次 flow draw 算各事件的 occupancy，再編碼成邊際 bins。Counts 乘條件均值後求和給該事件的 noiseless total。共享 flow 保留由 occupancy 造成的 totals 關聯，但不能把 bins 還原成真實的 herald × terminal 狀態對。Raw／trace 沿同一發的 chain 取樣，不配回 compact bin。實驗的 mask、分類與 reducer 仍用實際抽到的單發 IQ，見 [ADR-0074](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)。

## 問題

原本每次投影會依 DC 態分岔，而且每條 history 帶入自己的腔場。即使兩條 history 在同一能階，帶入的場不同，下一段仍各算一列。在 `chain/amp_rabi` 的首點，一次診斷累計 4,346 個 history snapshots 與 8,549 次腔場 advance。既有 [ADC span memo](2026-11-21-projective-span-reuse.md) 只重用完全相同的場求解，不能消除 history 軸。

使用者 2026-11-21 提出：「我自己估计的事情是，由于每一次中间的演化，也就是两次 real 间的演化都长得一样，那其实只要算一次演化的转移矩阵，就可以完成模拟了。」「既然腔場礙事，那就每次開始都丟棄腔場如何？」清空腔場改的是模型；改用 chain 連取樣表示也一起改。審查要求分開驗證兩步，不能以只丟場的耗時當作 chain 的加速，也不能以歷史剪枝直接解釋新 floor。

## 考慮過的選項與比較

- **R-a，只丟場。** 不改 history 表示，先量新初態的影響。四組早期樣本與同 commit 的保留場結果相比，最大終態 density 差為 `4.2e-9`，最大讀出 population 差為 `5.0e-9`。這是模型改動的局部對照，不是獨立精度。
- **R-d，丟場後同層去重。** 同點、同層、同 DC 態只解一次再分給各 history，保留原 compact joint。可省求解列，無法省掉 lineage、剪枝與 history 取樣。
- **R-b，丟場後改用 Markov chain。** Rank-1 投影固定 density，而真空重設固定場，所以下一段由層、點和 DC 態決定。不要求跨層共用一張矩陣。共用 ADC 既有的逐態 flow 與 raw 取樣，無需新增表示或分配 history 預算。Floor 依逐點來源機率判斷；在同模型內 floor 為零時，chain 與枚舉路徑給出同一分佈，但相同 seed 不保證抽到相同 shot 序列。

## 被否決的選項與理由

- 保留帶場 histories 或逐位合併。前者維持瓶頸，後者在 floor `1e-5` 的樣本裡至多減到約 1／1.8 的列數；floor `1e-9` 的測試前綴沒有可合併的場。
- 按場容差合併，或保留最近幾次 outcome。前者缺少從場差到最終誤差的界。後者要讓 ADC 認得額外的節點軸，而四組丟場對照沒有顯示保留記憶的必要。跨週期快取也沒有納入；短表的收益未證，不能只憑絕對 phase 不同就判定無法重用。
- 為 compact 保留舊 joint 或另加指定事件對。舊 joint 會保留 history 成本，新增事件對也不能替代 IQ 分類。未找到需要 compact latent-state joint 的 repo 呼叫端；這不排除外部呼叫端。
- 修正已知例外 R5。ADC 的 event 原本就從真空起算。只改量子續演化不會移除這個 IQ 誤差。在 `chain/t1` fixture，零間隔的 IQ 差是 g/e 間距的 3.75%，間隔 `5/κ` 時是 0.328%。使用者 2026-11-21 答「A」，接受保留該近似；依前後態計算 IQ 留待有需求再做。測試條件見 [FTF 已知近似](../../lib/zcu_tools/simulators/ftf/environment/README.md#known-approximations)。

## 量到什麼，還缺什麼

- 在 `chain/amp_rabi` 首點、heralded conditional phase 縮樣、`chain_power` 0.2 a.u. 與 `chain_length` 0.2 µs，候選 floor `1e-5` 對舊真空路徑的最大終態 density 差為 `2.682e-3`，最大逐讀出 population 差為 `3.272e-3`。同候選兩個 floor 的對照最大差小於 `7e-6`。以舊路最終保留的 histories 回算首讀，才出現 `2.758e-3` 的差；剪枝前的首讀沒有差。這只確定首讀的差源。後段與終態未完整歸因，因此沒有用候選 floor 0 取代舊路作遷移基準。兩種對照均過新 `1e-2` 門檻，均不是獨立參考誤差。
- 獨立參考以同一真空模型為準。兩個 floor 都通過的有 41／75 個 key：兩個 readout 優化縮樣各 17 個，加上 heralded conditional phase 的 7 個首讀。另 34 個未獨立驗證。當次參考合計 3,575.11 秒，含先前探針。Driven 列的解算成本高，且已有粗細解不足以封住場與積分誤差；未解和未有誤差界的 key 不算通過。
- 使用者 2026-11-22 同意先交付、缺證另補。只有一個共同 root seed，舊 heralded conditional phase 缺可配對的歸約前 compact 均值與總量，分類判讀規則也未凍結。分類 counts 保持封存，跨版本分類和 compact 統計未驗，舊 history 路徑仍保留。這些縮樣也不能證明四案的正常完整 Run 都符合 180 秒。

## 依據

Frontmatter 分別記早期成因調查、凍結的遷移與參考候選，以及交付前的程式版本。原始工作檔未入庫；本文保留判讀所需的數字與限制。驗證時序的使用者裁決見[模擬器的時間門檻與驗證成本](2026-11-22-simulator-speed-first-verification.md)。
