---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-17-pareto-selection-and-scene-append.md
  - 2026-11-15-single-segment-benchmark-boundary.md
  - ../../tools/README.md
basis:
  repo: 2514d94bef8b6500e6cc3f4a4bbf4663616d7001
  external: none
---

# DOP853 profile 場景的單列選取與 producer 接線

## 結論

本輪由 `tools.dop853_segment` 持有 profile 座標與座標衍生名稱，公開 `profile_inputs(scenario)` 單列選取。
Producer 的 `tools.family_pareto_scenarios` 接到這個 owner，不另存一份 profile 名稱或座標表。
選取先於 Input factory，未知名稱在 factory 前失敗，列出合法 profile 名稱。
預設 producer inventory 在固定 scans 後附加 profile 宣告，Pareto 預設選集仍維持原有 19 個名稱。
既有分工仍由 [ADR-0072](../adr/0072-module-boundaries.md#開發工具的數值交接) 與 [ADR-0078](../adr/0078-evolution-module-boundary.md#凍結數值輸入與參考解交接) 擁有，本輪沒有新增架構規則。

## 問題

既有共用清單已允許 profile 列與 Pareto 列共存，但 producer 還需要取得 profile 宣告並按名稱路由。
Profile 座標在專用 module，若 producer 自行重列座標或解析名稱，兩邊都會成為場景定義的來源。
若先建完整 catalogue 再挑一列，單列準備也會建構未選模型，未知名稱還可能先遇到無關的建模錯誤。

本輪只補這個公開入口與接線的討論紀錄。
原始工作檔未入庫，必要事實寫在本文，公開契約見 [工具用法](../../tools/README.md#dedicated-dop853-segments)。

## 考慮過的選項與比較

完整 catalogue 建好後再篩選，可以沿用無參數入口，但單列請求仍承擔其他模型的建構成本與失敗。
由 producer 維護名稱到座標的對照表，可以直接路由，卻要同步維護 profile owner 的宣告。
讓 `profile_inputs(scenario)` 在座標宣告中先選名稱，再呼叫單一 Input factory，則名稱、座標與錯誤都留在同一個 owner。
無參數呼叫仍回傳完整 catalogue，不改原本的整批用途。

本輪採最後一項。
`make_real_inputs(scenario)` 保留固定 scan 的直接路由，其他名稱交給 profile owner。
Producer 的 `main(argv, inputs=...)` 沿用既有介面，仍接受 caller 提供的宣告，也沿用 `--scenario NAME`，不加新 selector 或旗標。

## 被否決的選項與理由

不在 producer 複製 profile 座標或名稱規則，避免改一個 profile 時必須同步兩份宣告。
不以建好全部 Input 後篩選代替先選取，避免未選模型影響單列請求。
不因 producer inventory 擴充就擴大 Pareto 預設集合，因為 profile 列不是 family 去留證據。
不讓接線自動更新 index、填 scene cache 或求解 reference，因為宣告可建構的輸入不等於核可執行數值準備。
這些準備仍須明確操作與各自適用的授權。

## 留下的問題

接線本身不表示 index、cache 或 reference 已備妥，也不證明冷快取可以重建。
發布新列仍須透過既有 producer 明確指定 `--update-index --scenario NAME`，reference 準備仍由 tests 側 producer 負責。
本輪沒有改數值、worker 預設或量測方法，也沒有以接線宣稱效能收益。

## 依據

- 遵循[共用清單的 Pareto 選集與明確新增場景](2026-11-17-pareto-selection-and-scene-append.md)的選集與發布契約。區別在於該篇處理共用選取與新增列，本輪補 profile owner 的公開單列入口及 producer 路由，不把先例當作已記錄本次接線。
- 遵循[單段雙 tone benchmark 的量測邊界](2026-11-15-single-segment-benchmark-boundary.md)的專用 Engine input 與模型 owner 分工。區別在於該篇處理單段構造與量測邊界，本輪只接宣告來源，不新增 acquire workload，也不改 reference key。
- 遵循 ADR-0072 的工具與計時 owner 分工，以及 ADR-0078 的 solver-free codec 與 tests 側獨立求解。本輪沒有把場景選取搬進計時 owner 或 codec。
- [Profile owner](../../tools/dop853_segment.py) 的 `profile_inputs` 與 [producer](../../tools/family_pareto_scenarios.py) 的 `make_real_inputs`、`main` 擁有公開介面。[工具用法](../../tools/README.md#dedicated-dop853-segments)保存呼叫方式與準備邊界。
