---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/draft/simulation-environments.md
  - ../../lib/zcu_tools/program/v2/simulation/README.md
basis:
  repo: 1b1db6c13d4538be1f14ff7f8651c5f4eeb56320
---

# Acquire outcome 的作用域與編譯交接

## 結論

採用 binding-owned acquire 作用域，刪除分開呼叫的 begin/end，不留相容入口。
Mixin 把 builder 放進 compiling 作用域，把 request release 與 active 清理留在 acquire 本體內。
Environment 收同一個 frozen `AcquireOutcome`，不另外定義鏡像結果型別。
現行介面與例外契約由 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md) 擁有。

## 問題

分開的 begin/end 沒有表示 acquire 本體是否正常返回。
Observer 需要區分編譯、執行與清理拋錯，還需要保留 builder 拋錯前已花的秒數。
Stop 可能正常返回部分結果，所以不能把「沒有做完全部 rounds」直接當成拋錯。

## 考慮過的選項與比較

- 由 program 把多個位置的成敗旗標傳給 end。這會把成敗判斷分散到 builder、round loop 與 cleanup。
- 在 end 讀外層例外狀態。成功 acquire 可能位於 caller 的 except 區塊，外層錯誤不是這次 acquire 的結果。
- 由 binding 自己的 with 本體判斷。這個範圍能同時涵蓋 builder、執行與 mixin cleanup，observer 不必推測 caller 狀態。

採用第三種。Complete 表示正常返回，不表示預定 rounds 全部完成。
Observer 一般 Exception 只 warning 一次，不改返回或遮蔽原錯；BaseException 沿既有行為外傳並保留原錯 context。

## 被否決的選項與理由

不引入 handle、Protocol、collector 或 environment 基類。
既有 binding 已擁有 admission 與 round 輸入，新增一層不會移走這些責任。
Program 不 import telemetry writer，environment 只投影 outcome 到既有 summary。

遵循 [ADR-0083](../adr/0083-program-macros-and-timelines.md#compiled-snapshot-與-simulation-入口) 的先例：compiler 擁有編譯產物，simulation 不接管 compile lifecycle。
這次只量測 request builder，不把 install 或後續執行算成編譯。
沿用 [ADR-0066](../adr/0066-operation-lifecycle.md#取消與互動) 的 Stop 邊界，不另加「全部完成」判斷。

## 留下的問題

Evolve 與 readout 兩段計時尚未接線，FTF summary 暫留未量測值。
後續 environment observer 沿用 AcquireOutcome，不重新判斷 complete。

## 依據

本次實作依據已核准的 acquire lifecycle 契約；原始工作規格未入庫。
介面責任沿用 [模擬環境草稿](../adr/draft/simulation-environments.md#acquire-接線與-projective-readout) 與上述 ADR。
