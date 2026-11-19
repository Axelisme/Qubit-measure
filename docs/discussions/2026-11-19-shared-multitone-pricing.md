---
date: 2026-11-19
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-11-18-sampled-multitone-rwa-admission.md
  - docs/discussions/2026-11-18-gate-direct-duration-price.md
basis:
  - "full-drive gate lane@0e28cba81de800a7d53e93da02f4bfeda71d753f"
  - "sampled RWA main@ea668f77cd2b8e654808d457b0fb767004c6535c"
---

# 成形雙 tone 的共同選路

## 結論

成形雙 tone 同時有 gate 和 sampled RWA 候選時，讓 dispatch 按群總 CPU 秒選先驗的方案。比較項包括 RWA 的整群探針與播放、gate 的建構與播放，以及不使用兩者時的原路由。只在兩個候選可能重疊、且原路由不是合格 GPU 波次時換尺。原先的門檻、solver 和輸出不變。現行規則見 [evolution 的決策圖](../../lib/zcu_tools/simulate/evolution/README.md#decision-graph)。

## 問題

接合前，gate 用本次 evolve 的 key 使用次數攤建構。Sampled RWA 用振幅群攤抽查，再和 worker 並行成本或合格 GPU 波次比。兩套價格未排成同一序列，哪個 guard 先跑取決於接合位置。

RWA 群比對實際載波相位，因為相位改變時不能借用另一群的探針。Gate key 不含相位，Fourier 播放吸收相位差。同一個 key 因此可能跨群，連同不入群的列共用一次冷建構。

## 考慮過的選項與比較

選擇按群分攤建構，先驗計畫選 RWA 的群，再按剩餘使用次數重做 gate 的回本檢查。假設一個 key 建構 40 秒、兩個相位群各用 50 次，原路由每次 0.5 秒，忽略 gate 播放。兩群都走 gate 時共付 40 秒，比原路由的 50 秒省 10 秒。一群走 RWA 後，剩餘 50 次只省 25 秒，不能獨付 40 秒，於是不建構。

這遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md#內部允許實質耦合dispatch-擁有決策知識)：dispatch 擁有適用條件、價格與候選順序，kernel 保有數值 guard。也保留 [sampled RWA](2026-11-18-sampled-multitone-rwa-admission.md) 的逐群探針與前置成本，以及 [gate 的 CPU 對照估價](2026-11-18-gate-direct-duration-price.md) 的原校準。白名單排除的 family 不參加比較。單一 pin 可以跳過價格，不能跳過 guard。

使用者於 2026-11-18 將 family 納入與做法的選擇交給 designer：「要不要纳入，要不要采取变种，是你们帮我决定的。」同日對既有 RWA 取捨說：「以比较快的场景稍微变慢一点为代价，让慢场景快很多是可以接受的，所以可以采用你的建议。」兩段原話的出處見 [sampled RWA 實測放行](2026-11-18-sampled-multitone-rwa-admission.md#考慮過的選項與比較)。Designer 依此選定分攤與重核，不改數值 guard。

## 被否決的選項與理由

- 固定先驗 RWA 或 gate。探針通常比建構便宜，但價格改變或重用增多時，固定順序會先付一筆無法回本的 guard。
- 每群都收整份 gate 建構，或只分攤不重核。前者丟掉跨相位群的重用，後者讓選了 gate 的群承擔原本分給 RWA 消費者的成本。
- 將共用 gate key 的群合成一個最佳化問題。這要搜尋各群的混合配置，還需要新的局部拒收規則。現有量測沒有這類工作負載。
- 將所有 sampled 群改用 CPU 秒。沒有 gate 候選時會更動已驗收的選路，CPU 秒也不能直接和 GPU wall 秒相比。

## 留下的問題

共尺可能使部分短場景多付一次 RWA 探針。這次用既有正式場景的逐列選路核對，不以查詢時間代替端到端計時。CPU gate 與 GPU 的共同估價尚未建立。

## 依據

[公開路由測試](../../tests/simulate/evolution/contract/test_shared_multitone_routing.py)使用合成雙 tone 驗證先驗、拒收回退和跨相位群的重核。原決策調查工作檔未入庫。可重現的規則由 [dispatch](../../lib/zcu_tools/simulate/evolution/_shared_multitone_dispatch.py) 擁有，數值放行仍由 [kernel](../../lib/zcu_tools/simulate/evolution/kernels/README.md) 擁有。
