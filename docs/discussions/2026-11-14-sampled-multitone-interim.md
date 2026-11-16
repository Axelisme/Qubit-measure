---
date: 2026-11-14
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
basis:
  - "repo base@4afda3b4172ed98e5b9878be5bb0022c307bbe67"
---

# Sampled multitone 的過渡期拒收

## 結論

這次暫停 `multitone_rwa` 與 `multitone_gate` 的 sampled-envelope admission。`multitone_gate` 保留原 family 名稱與數值 builder，但不再接收列。`multitone_rwa` 保留 distinct-port constant pairs 的現有資格與數值實作。使用者於 2026-11-14 對「過渡期先用 light change 進 main，另外兩件開成新 task」選「A」。

現行輸入、fallback 與暫停條件由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 擁有。這次不修改 constant-tone 的 omission guard、endpoint kick、common-frame 定價或 single-tone 的共享數值規則。

## 問題

Sampled multitone 的 omission proxies 不能替完整 pulse sequence 的最終 complex density 背書。保留所有 sampled admission，會讓尚未取得完整序列精度資格的路徑繼續被選上。這次維持最終 density 最大絕對差 `2e-3` 的驗收門檻，不把局部 budget 或單段通過當成整條序列合格。

過渡期有速度代價。拒收後改由其他候選計算，不能用使用者目前較少使用這類序列的判斷取代 benchmark。完整 benchmark 的任一 workload、history mode 或 clock 變慢超過 10%，仍須在合併前回報使用者。一次性研究工作檔未入庫，本篇不把研究計時當成正式 benchmark 結果。

## 考慮過的選項與比較

維持現狀可保留 sampled 快速路徑，但沒有解決完整序列的精度問題。只拒收串接序列、保留較弱單列，需要另定 admission 門檻與 held-out 驗證。先完成新的 sampled 修法再一起落地，則會延後這次已裁定的拒收。

採用過渡期拒收，讓既有 dispatch 繼續比較其餘候選。先前屬於 cheap RWA 的 sampled 列會失去 common-frame 價格豁免。近載波列可能改選 `common_frame`，成本較高的列則保留 `dop853` fallback。因此不能把拒收實作成「所有 sampled 列強制選 DOP853」。

## 被否決的選項與理由

這次沒有採用弱耦合單列豁免，因為它需要新的數值界線與驗證。也沒有刪除 `multitone_gate`，因為使用者選的是暫停，不是 family 去留。把暫停放入 single-tone 共用 admission 會擴大行為變更，故拒收留在兩個 multitone family guard。

## 留下的問題

Sampled 修法與 constant-tone guard 的後續設計不屬於這次變更。恢復 sampled admission 時，須同步恢復原本 19／20 次成本邊界的驗收。暫停期間，inventory 達到 20 次與 pinned allowlist 都不能恢復 admission。

這次遵循 [ADR-0078 的 family 白名單契約](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)。Pin 可以略過成本，不能略過數值拒絕。也遵循 [完整序列精度原則](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時)。本次沒有 family 刪除，也不重新定價其他 family。

## 依據

[Routing boundaries](../../tests/simulate/evolution/contract/test_routing_boundaries.py) 驗證 19／20 與 pinned 的暫停行為，保留 30 MHz 契約。[Accuracy matrix](../../tests/simulate/evolution/contract/test_accuracy_matrix.py) 對 active family 與 fallback 的完整 timeline 比較獨立 full-Hamiltonian Lindblad 參考解。它保留 paused family 的未執行狀態，不填零誤差。

候選順序與定價分別由 [multitone dispatch](../../lib/zcu_tools/simulate/evolution/_multitone_dispatch.py) 與 [common-frame dispatch](../../lib/zcu_tools/simulate/evolution/_common_frame_dispatch.py) 擁有。這次沿用 [common-frame 成本歸 dispatch 的裁決](2026-11-14-common-frame-step-pricing.md)，不把性能出口寫成新的數值 guard。

## 後記，2026-11-15

使用者於 2026-11-15 追認 designer 的拒收終態與下述合併驗收方法，原話為「可以追认然后把优化任务安排下去。」`multitone_gate` 去留與 STIRAP 基準採縮短 Gaussian 的依據仍另問。這項變更不預告修好後恢復 sampled admission。`multitone_rwa` 只收既有 distinct-port constant pairs，`multitone_gate` 保留但拒收所有列。19／20 與 pinned 契約繼續驗證拒收。其餘候選沿用原定價，包含 cheap nearby 的 `common_frame` 與 separated 的 `dop853`。現行契約仍由 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary) 擁有，完整 density 標準與既有 strict xfail 不變。

本次合併驗收的方法也改為單調性論證加公開 telemetry，而不是原先的 exclusive 計時。Guard 只收窄候選，因此 main 沒有把 sampled 多 tone 列交給這兩個 family 時，收窄不會改變選到的 family。四個非 overlap benchmark workload 由版本化設定、program 序列與 domain 分類確認不受影響。`ftf_overlap_shaped` 則由 `run_workload` 在 main 完整執行三個 mode，每側 19 次 acquire，包含暖機。Main 每次 acquire 的兩個 multitone family 選中 rows 均須為零，after 用同一驅動程式重播作佐證。這裡描述驗收方法，不代表量測已完成。

此方法不以 family 彙總相等冒充逐 row 比對，也不把非 exclusive 計時當作正式 slowdown 數字。資料缺失、前提不成立或兩側不一致時，合併前停止，再提交所需 exclusive benchmark 的方法與作業時間額度。使用者已追認這項對原完整 benchmark 方法的區別。STIRAP 的本機模型未入庫，其選路與耗時推論不列入本次合併門檻；沒有 after 讀數的項目仍是推論。

使用者於 2026-11-15 解除 4-core 上限並重新允許 exclusive timing。這不等於核准任何特定作業，也不重置既有 once／retry 額度。本次 gates、formatter、commit 與觀察仍須另授，不能因 CPU 解鎖而啟動舊申請。

## 後記，acquire benchmark 退休

入庫的 program → acquire 端到端 benchmark 已退休，上述 `run_workload` 僅記錄當時的驗收方法，不再提供執行入口。整體耗時改在 `Engine.evolve` 的多段序列層量測，見 [工具說明](../../tools/README.md#simulator-benchmark)。
