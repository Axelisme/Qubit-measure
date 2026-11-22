---
date: 2026-11-23
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0084-simulation-integer-time-base.md
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-21-raw-adc-word-reject.md
basis:
  scan_commit: c15a15706e064009c9f620be92ac6f2948fe6806
  main_base: 2d616c9ef7aed7ac5bfb9f4c3f6fbb6058c3ebd8
  external_sources: none
---

# Program 的模擬邏輯只在 mixin 內

## 結論

Designer 依 2026-11-23 的授權代裁，待使用者追認。整數 request 的單輪估時搬進 `program.v2.simulation`，保留共用的 `shot_time_s`。STIRAP 不再沿取數介面傳送模擬專用的線路要求。新的 import contract 把 Program 對模擬套件的直接依賴限制在承載 mixin 的子包，不放寬既有的全框架禁令。現行規則見 [ADR-0072〈真實 acquire 與模擬邊界〉](../adr/0072-module-boundaries.md#真實-acquire-與模擬邊界)與 [ADR-0084〈模擬 round 時長〉](../adr/0084-simulation-integer-time-base.md#模擬-round-時長)。

## 問題

2026-11-22，使用者：「据我所知啦，模拟器这个功能啊，应该只局限在一个 mixin 内，也就是说，如果把这个 mixin 拿掉，我的 program 就没有模拟功能。这个是解耦的设计。」「我认为这两处可能有违背了我的设计理念。对我来说，只要把这个 mixin 拿掉，我的 program 就不会有模拟的逻辑存在了。」

當時掃描到兩處會留在 mixin 外的模擬專用行為或介面：

1. `program/shot_time.py` 的 `request_round_time_s` 依賴 `simulate.boundary.AcquireRequest`，也從 `zcu_tools.program` 公開匯出。只有模擬 binding 用它做 round pacing 與時鐘補時。共用的 `shot_time_s` 則是八個 workflow 使用的 µs 估時入口。
2. STIRAP 的 amplitude、delay、detuning 三個實驗，經 `Schedule.run_acquire_summed` 把模擬專用的 charge 線路要求交給 Program acquire，mixin 再經 compiled builder 把它寫入 request。Single-fluxonium 用它核對 generator 是否接到 charge 線路；FTF 和 Six 不支援該檢查，一收到非空要求就拒絕。這些要求沒有增加用於演化的脈衝或接線資訊。

## 考慮過的選項與比較

- 從編譯 metadata 或 pulse tag 推導 charge 角色。lab 可省掉 acquire 參數，Single-fluxonium 仍能核對角色，但 Program IR 必須記住穩定角色，builder 和環境再讀回。只猜 pulse 名稱無法可靠地完成這個檢查。
- 由外掛 hook 把線路要求注入 binding。一般 acquire 可移除專用參數，lab 仍得組同一份設定，環境也仍得決定如何處理。
- 保留現狀。沒有介面改動，也保留 Single-fluxonium 的額外接線檢查，但移除 mixin 後，Program 公開介面與共用估時仍留下模擬專用部分。

這次刪除了整條取數鏈上的專用參數與 request 欄位。環境繼續依接線與已編譯的事件運算。代價是 Single-fluxonium 不再判斷作者是否選錯線路類型。FTF、Six 不再因收到不支援的線路要求而拒絕，STIRAP 可以在 FTF 上照算。這不保證作者的接線正確。單輪估時只換 owner，不改公式，不留 Program 的相容匯出。

## 被否決的選項與理由

Metadata 方案把只供驗證的角色搬進 IR，讓多個 package 為同一項檢查維護新契約。Hook 只把專用設定搬出 acquire，lab 仍要供應它。都不符合工具只做自身職責的取捨。保留現狀則直接違反使用者提出的 mixin 邊界。

兩個被換掉的決定是 [ADR-0072](../adr/0072-module-boundaries.md) 原先傳遞線路要求的規則，以及 [ADR-0084](../adr/0084-simulation-integer-time-base.md) 原先把單輪估時放在共用 Program 模組的決定。兩者分別是 2026-10-30 與 2026-10-31 的 advisor 代裁，未見使用者覆核。本次遵循 [ADR-0078〈模擬器不替使用者擋實驗參數〉](../adr/0078-evolution-module-boundary.md#模擬器不替使用者擋實驗參數)：合法輸入照算，已知模型近似可用警告說明。與 [raw ADC 頻率字編譯時拒絕](2026-11-21-raw-adc-word-reject.md) 的區別在於，raw 字缺少決定物理頻率的資訊；STIRAP 的線路要求只用於驗證，拿掉仍有完整的運算輸入。

## 留下的問題

模擬環境的 generator／ADC 路由驗證與 `SimulationSoc.install_request` 的路由綁定檢查不變。Experiment 的 [Run 來源紀錄](../../lib/zcu_tools/experiment/acquisition.py)使用 `SimulationSoc` 來填 metadata，不是 Program 取數。Lab 使用 [`simulate.fluxonium` 的擬合與 predictor](../../lib/zcu_tools/simulate/fluxonium/README.md)，用的是物理模型而非模擬取數 runtime。兩者不在這次裁決範圍。

使用者尚未追認代裁，也未核准 ADR-0072 裡的原則草稿措辭。

## 依據

掃描以 `c15a15706` 的 Program、Schedule 與 STIRAP 呼叫點為準。上文列出辨識出的兩處依賴與未改範圍。原始掃描筆記及使用者討論未入庫。
