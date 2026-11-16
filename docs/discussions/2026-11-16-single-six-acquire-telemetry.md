---
date: 2026-11-16
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
  - ../../lib/zcu_tools/simulators/README.md
basis:
  repo: 21810d02064a09ca58140c6b8fda101690d144d1
  external: none
---

# Single 與 six 的 acquire telemetry

## 結論

Single 與 six 接上共用 `RoutingTelemetry`，每次 admitted acquire 寫一行。
設定使用共用 `TelemetrySettings`，預設開啟。
Six 先聚合所有 source、routed group 與 projective span，再寫入一次。
兩者直接讀 binding 的 `AcquireOutcome.compile_wall_s` 與 `complete`。
正常返回的 partial Stop 是 true，數值拋錯是 false。

## 問題

Single 與 six 原本沒有 acquire telemetry。
Single 的 projective readout 會多次呼叫 Engine，six 又會把一個 request 分給多個 source。
若在每次 Engine 返回時寫檔，同一次 acquire 就會產生多行，也無法表達後續失敗。

## 考慮過的選項與比較

採用 source 內的 acquire 累計值，在 admission 時歸零。
每次 Engine 呼叫開啟 telemetry，呼叫外的 finally 累加 wall 秒數。
Projective runner 保留已完成 span 的診斷，讓後續 continuation 失敗時仍能交給 source。
ADC preparation 與取樣另計 readout 秒數。
未進入的段維持 `0.0`，延後的 raw materialization 不算入 acquire 行。

遵循 [共用 writer 先例](2026-11-16-shared-acquire-telemetry.md)，沿用八欄 summary、bounded JSONL 與 warning-once I/O 隔離。
遵循 [Acquire outcome 先例](2026-11-16-acquire-outcome-scope.md)，environment 不重判成敗。
遵循 ADR-0078，只讀 Engine 的公開 `PathDiagnostics`。
Complete preparation 重用透過公開 `for_reuse` 產生當次零 CPU 診斷，不重複計入上次求解的 CPU。

## 被否決的選項與理由

不在各個 source 或 Engine 呼叫處寫行，因為 six 的 acquire 才是統計單位。
不加 collector、Protocol 或 environment 基類，兩個 owner 各自持有計數即可。
不讓 environment 以 round 數推斷 complete，因為 partial Stop 可能正常返回。
不把段欄位重新宣告在 child summary，共用型別仍是唯一宣告處。

## 留下的問題

共用三段 annotation 仍允許 null，供其他呼叫端的過渡接線使用。
Single 與 six 已填 float，型別收緊與 analyzer 彙總由後續改動處理。
Telemetry 開關控制 JSONL 寫入，這次仍對每次 single/six evolve 開啟 Engine 診斷。

## 依據

現行契約由 [simulators README](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 與兩個 environment README 擁有。
模組定位依 [ADR-0072](../adr/0072-module-boundaries.md)，Engine 黑盒與重用依 [ADR-0078](../adr/0078-evolution-module-boundary.md)。
原始任務規劃未入庫，本篇記錄採用方案、過渡邊界與被否決方案。
