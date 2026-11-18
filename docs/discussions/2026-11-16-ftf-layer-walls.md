---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - ../../lib/zcu_tools/simulators/README.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo: 21810d02064a09ca58140c6b8fda101690d144d1
---

# FTF acquire 的演化與讀出 wall

## 結論

FTF 在每次 Engine.evolve 與 ADC／讀出的實際呼叫處累積 wall。
同一次 acquire 的所有 rounds、腔記憶 spans 與 projective continuations 共用兩個加總值。
計時作用域用 finally 保留失敗中的時間，未進入的段回報 0.0。
Environment 繼續直接投影 binding 的 AcquireOutcome，不另判斷 complete。
現行欄位與範圍由 [FTF telemetry 契約](../../lib/zcu_tools/simulators/ftf/environment/README.md#routing-telemetry) 擁有。

## 問題

FTF 原先只回報總 wall、編譯 wall 與 family CPU。
總 wall 無法區分演化和讀出，family CPU 也不包含 ADC。
腔記憶與 projective readout 在演化呼叫之間求解腔場，不能把整個 continuation 函式都算成演化。
數值錯誤也可能發生在任一段，成功返回後才加總會漏掉失敗段。

## 考慮過的選項與比較

- 圈住整個 preparation。這會把規劃、腔場、演化與 ADC 混在同一段。
- 只圈住最外層 evolve_plans。這會把腔記憶中的讀出工作算成演化。
- 在呼叫處放計時作用域。現有 CavityEvolutionOptions 傳入兩個作用域，預設不觀察。FTFAcquire 持有加總，runtime 和 cavity 各自圈住其負責的數值工作。

採用第三種。兩個作用域不巢狀計入同一份工作，不改 solver、抽樣流或 reuse 判斷。
Preparation 與 pacing 留在總 wall 的餘數，acquire 後才要求的 raw materialization 不回寫已產生的行。

## 被否決的選項與理由

不加 collector、environment 基類或新的結果鏡像型別。
現有 acquire session 已擁有 round 的生命週期，現有 options 已把 continuation 控制送到每個 span。
計時不需要另一個持有者。

遵循 [ADR-0078](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)，environment 只在公開 Engine 與 ADC 邊界觀察工作，不進入 dispatch 或 family。
沿用 [Acquire outcome 的作用域與編譯交接](2026-11-16-acquire-outcome-scope.md)，binding 是 complete 與 compile_wall_s 的唯一來源。
沿用 [共用 acquire telemetry writer](2026-11-16-shared-acquire-telemetry.md) 的一次 acquire 一行與 I/O 隔離，不新增寫檔路徑。

## 留下的問題

其他 environment 的接線與共通三段型別收緊仍各自處理。
本次不改 analyzer 或擴充分段欄位。

## 依據

本次依據已核准的分層計時規格，原始工作規格未入庫。
驗收從公開 acquire/binding 觀察 JSONL，Engine 與 ADC 的觀察器委派真實小模型。
合成時鐘只在公開數值工作完成或拋錯時前進，用來檢查加總與段歸屬，不設真實耗時或速度門檻。
