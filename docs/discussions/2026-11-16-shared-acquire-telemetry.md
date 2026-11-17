---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulators/README.md
basis:
  repo: de8245e3ff4ff958a05663771bf15dc30bbfeaff
  external: none
---

# 共用 acquire telemetry writer

## 結論

FTF 的 bounded JSONL writer 與設定型別移到 `simulators.telemetry`。
`AcquireTelemetry` 集中呼叫端的八個必填欄位，`RoutingRecord` 繼承它，只加 writer 產生的四欄。
`write(summary, *, paths)` 明列欄位投影，不展開整份 summary，也不修改它。
FTF 的 projection 統計只寫入帶 `ftf.` 前綴的 `extras` key。
契約與使用方式由 [simulators README](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 擁有。

## 問題

原 writer 在 FTF environment 內，其他物理模擬器無法在不依賴 FTF 的情況下使用它。
逐一新增三段耗時、成敗與 extras 參數，也讓公開方法超過既有參數數量門檻。
各 environment 需要同一份資料契約，不需要共用 acquire 流程。

## 考慮過的選項與比較

- 保留多個 keyword 參數。呼叫端直接填值，但每加一個共通欄位就要擴張方法簽名。
- 只包三段耗時。可以縮短參數列，但成敗、extras 與基本計數仍散在方法參數中。
- 用完整 acquire summary。八欄在同一個公開型別宣告，writer 另行彙總 Engine 診斷。採用此案。

遵循 ADR-0072 的模組定位與單向依賴，writer 留在組合物理模擬器的中性層。
遵循 ADR-0078 的黑盒邊界，family 彙總只讀 `simulate.evolution` 公開診斷。
沿用原 writer 的鎖、檔名、50 MiB 上限與 warning-once 政策，不加 collector、Protocol 或環境基類。

## 被否決的選項與理由

不讓 child TypedDict 重宣告三段耗時，否則同一欄位會有兩份型別權威。
不使用 `NotRequired`，因為未量測也是呼叫端必須明確交代的狀態。
不使用 `**summary`，避免把 caller 擴充欄位混進持久格式。
不保留頂層 projection 雙軌欄位，避免 consumer 面對兩種位置。

## 留下的問題

這次搬移先允許三段耗時為 `None`。FTF 明填三段 `None` 與 `complete=True` 作暫時接線。
編譯耗時與成敗要由 program acquire observer 提供。各 environment 再補演化與讀出計時，完成後把三段型別收緊為 `float`。
現有 analyzer 只讀 `families`，沒有 projection consumer，因此不隨這次搬移改動。

## 依據

以 basis commit 的 FTF writer、公開 acquire 測試及 analyzer consumer 為準。
原始設計裁決未入庫，本篇保留採用的 API、過渡限制與被否決方案。
定位規則見 [ADR-0072](../adr/0072-module-boundaries.md)，Engine 接縫見 [ADR-0078](../adr/0078-evolution-module-boundary.md)。
