---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0084-simulation-integer-time-base.md
  - ../adr/retired/0077-numerical-plans-and-replay.md
basis:
  repo:
    main: d45e87637975cbbfffa566a8888635dbebe28834
    candidate: c6f6582f75f241a6745b4881f8757473eefd9548
---

# 移除 private replay

## 結論

2026-11-18，使用者選「A」，移除 FTF 與 single-fluxonium 的 private replay，不留讀取或轉換舊 archive 的入口。除錯改讀 debug capture manifest。它記下 Engine 輸入、每列實際 routes，以及 acquire 成敗、seed／sequence、每輪 DC、model identity 和有效設定。使用方式見 [模擬器的 debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)。

## 問題

開發動態維度時，逐位重算要求存下選到的維度，還一度迫使求解採用一致的形狀，把 pilot 門檻從 8 條 timeline 提到 16 條。數值改動還得更新 replay 版本和 metadata。這條路徑約有 560 行程式、740 行專用測試。GUI 與 `zcu_lab/` 都不呼叫它。

使用者指出成本放錯了地方：「关于 replay 啊之类的其他成本，拜托你以优化的效率为优先，而不是这些其他的细枝末节的仪式、一些储存格式。你如果迁就它的话，优化很难做啊。」

使用者接受把過程資訊存下來：「我自己认为，好，你有一个可以打开的 debug 模式，然后会把过程中的一些结果把它 dump 到一个资料夹或者一个档案里啊，这可以。但要求完全 replay 从自完整的算法就有点超过咯。呃，你需要的资讯应该本身就在 dump 的结果里了。如果你会需要 replace 多余的资讯、多余的算法，那就是你 dump 的资料不够多嘛。」

## 考慮過的選項與比較

| 選項 | 代價 |
| --- | --- |
| A：刪除兩個環境的 private replay，以 debug capture 補足判斷資訊 | 可以檢查當時的輸入和路由，不再保證用舊 archive 重算。也不必為重算改動求解。 |
| B：只刪 FTF 的 private replay | FTF 不再為 replay 改動求解，但 single-fluxonium 仍需要共用 archive 與版本核對。 |
| C：凍結 private replay，新優化碰到的 round 回報不支援 | 不必立刻改存檔，卻仍要判斷哪些數值改動讓舊 archive 無法重算。 |

ADR-0077 曾移除 numerical planner 與 checked plan replay。這次遵循同類做法：不用保留專供重算的格式來替代實際的診斷資訊。

## 被否決的選項與理由

使用者對三個選項答「A」。B 留下共用模組和 single-fluxonium 的版本核對。C 仍讓數值改動牽動 replay 的可用性。兩者都保留了這次要移除的成本。

## 留下的問題

動態維度所選的維度與 pilot 數據、sampled `multitone_rwa` 的抽查結果，由做出各項判斷的改動自行記錄。本次只補除錯所需的 acquire 與 round 資訊。

## 依據

檢視 main `d45e87637` 上仍在使用的 replay 入口，以及移除後的候選 `c6f6582f7`。原始工作檔未入庫。模組定位見 [ADR-0072](../adr/0072-module-boundaries.md#模組定位)；整數時間契約見 [ADR-0084](../adr/0084-simulation-integer-time-base.md)。同類先例見 [退役的 ADR-0077](../adr/retired/0077-numerical-plans-and-replay.md)。
