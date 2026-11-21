---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0083-program-macros-and-timelines.md
  - ../adr/0073-measure-form-derivation.md
  - ../../zcu_lab/v2/twotone/ro_optimize/chain_length/guide.md
basis:
  repo_commit: 32c7db0b57a71d503d2eca6ef29430416e9654cb
  external_sources: none
---

# 讀出鏈的 pulse 長度與間隔

## 結論

`chain_length` 掃 ADC 開窗時間。每點的 pulse 長度是 trigger offset 減訊號飛行時間，加當點的開窗時間與 0.11 µs。脈衝抵達 ADC 的尾端會在關窗後留下 0.11 µs。要覆蓋整個 window，trigger offset 須至少等於 pre-delay 加飛行時間。這是 guide 的物理說明，不是 Run 的拒絕條件。

表單開啟與 Reset 時擷取 resonator 的飛行時間，在唯讀必填欄位顯示。Refresh 保留這個初值。缺值或無效時，操作者先修正校準再 Reset。新增必要欄位使實驗的 cfg 從 4.0 升到 5.0。舊結果的 moments 仍能載入分析，舊 cfg 不還原。

九個 chain 實驗在前一個讀出 pulse 結束並等完腔衰減時間後，就能開始下一個步驟，不另等 ADC 關窗。原有的驅動、前置等待與 shot 邊界不變。編譯器仍檢查同一 ADC 的 trigger 不重疊。時序 owner 見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)，實驗行為見 [chain_length guide](../../zcu_lab/v2/twotone/ro_optimize/chain_length/guide.md)。

## 問題

舊公式把 pulse 設為開窗時間加 0.11 µs。[FTF fixture](../../tools/data/experiment_wall_time/) 的 R1 pulse module 在 1.111 µs 觸發 ADC，飛行時間是 0.387 µs。照舊公式，0.2、0.4、0.6 µs 的前三個 pulse 傳到 ADC 後，都在 ADC 開窗前結束。較長的 pulse 也只蓋住部分窗口。FTF 拒絕了第一點，實機即使完成也會把 pulse 未蓋滿的效果混進分數。牆鐘調查的原始工作檔未入庫，以上門檻可由 fixture 的兩個校準值與舊公式重算。

另外，共用的 readout module 已在發射結束後加 `post_delay`。ADC window 記在另一條 timeline。實驗在每次讀出後呼叫預設的 `delay_auto()`，才把 ADC 的尾端加進等待。

## 考慮過的選項

長度可以用 trigger offset 加開窗時間，也可以再減飛行時間。減掉飛行時間，讓 pulse 送到 ADC 的尾端落在關窗附近；沿用 0.11 µs 可以留出緩衝。間隔可以只修 `chain_length`、修九個 chain 實驗，或改共用 `delay_auto()`。三者的差別是有多少實驗要承擔行為變更。

使用者在 2026-11-21 裁定：「B''，ii，應該改的是readout module計算timeline的方式，不join ro window，然後實驗cfg要多預留先緩衝pulse長度，不要對齊到極限。」核對實作後，readout module 已不 join ADC，所以移除九個實驗各自加回的等待即可。這遵循 ADR-0083，不改 owner。飛行時間沿用 [2026-10-17 的初值規則](2026-10-17-measure-form-initial-values-and-sweep-mode.md)，把缺值、無效值與來源故障分開處理。固定候選的審查發現，隱藏欄位不在 Refresh 保留的輸入裡，只按 Refresh 就會換掉飛行時間。designer 在 2026-11-21 選用 [ADR-0073](../adr/0073-measure-form-derivation.md) 已有的唯讀初值規則，讓操作者看得到 Run 會用的飛行時間。

## 被否決的選項

- 不扣飛行時間的長度會多送一段相同時長的驅動。
- 只修 `chain_length` 會讓另外八個 chain 實驗繼續等 ADC 關窗。
- 改 `delay_auto()` 的預設會改到非 chain 實驗，即使那些實驗需要等 ADC。
- 隱藏飛行時間會讓缺少的必要校準無法直接從欄位辨認。允許手填則讓 Run 使用另填的飛行時間，不再反映已擷取的校準。

## 留下的問題

R1 fixture 的 `post_delay` 是 0 µs。新間隔下，前一個 ADC window 最後約 0.277 µs 會跨到下一個 pulse 的發射區塊。同 ADC 的 trigger 仍不重疊，但 FTF 當時要求所有 ADC samples 早於下一個 block。單次驗收沿用 fixture 的校準與模型，只將可拋棄的輸入副本設為 2 µs。Run 在 0.2 µs 的第一個窗口仍遭拒絕。當時的路由檢查先找與 ADC window 同時活動的 generator，未把飛行時間算進交集。該 pulse 的發射提早約 77 ns 結束，但抵達 ADC 的尾端仍晚於關窗 0.11 µs。這兩個模擬器限制都留待後續處理。入庫的 fixture 與 FTF 路由不改。

## 依據

- [FTF fixture](../../tools/data/experiment_wall_time/) 的 `ftf-demo/initial`，別名指向 R1 的 pulse module。2 µs 是該 point 的 MG12 mux module 設定，不是 R1 原值。
- [ADR-0083](../adr/0083-program-macros-and-timelines.md) 的 readout tail 與 ADC overlap 規則。
- [實驗離線環境契約](../../lib/zcu_tools/experiment/README.md#explicit-simulation-environments) 記錄 FTF 使用的 block 檢查。
