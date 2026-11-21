---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../adr/0066-operation-lifecycle.md
basis:
  - "本 repo 候選 a3d1e64023cfc0370d8bd370f24ae3656c0c4b85"
  - "QICK 0.2.436，qick.qick_asm.AcquireMixin.finish_round"
---

# FTF Stop：合作式檢查點與取消傳輸

## 結論

FTF Stop 用合作式檢查點，不設強制上限。使用者答「3C，但設計上5秒內」：5 秒是正常參數下放置相鄰檢查點的依據，不是 gate。FTF 在逐 outcome 的 projective advance 前查取消，Engine 在幾何規劃結束後也查一次。設計目標尚未達成，剩餘區段由 `ftf-stop-remaining-gaps` 承接。規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md)。

模擬 SoC 在 request 生命週期持有同一取消查詢，兩個 QICK round 入口都把它帶到 backend 與 pacing。Backend 中斷時只保留已完成輪。零完成輪由 mixin 拋 `StoppedPartialAcquireError`，不回傳空輪平均。這個接線和零輪處置是 designer 代裁，使用者未覆核。責任分界見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。

## 問題

原來的 `chain/amp_rabi` 在 GUI 送出 Stop 後，24.9 至 34.2 秒才到 Controller 終態或 idle。FTF 的 projective 組裝區段耗時 11.8 至 95.3 秒，逐 history 推進期間不查取消。Decimated 有另一個缺口：取消只能打斷 pacing，沒有傳入 FTF 求解。QICK 0.2.436 的 decimated round 走 `start_tproc()`，只補 `start_readout()` 到不了這個 caller。

初審另查到：注入的 backend 可以在 caller 旗標未設時拋 `Cancelled`。若 mixin 只結束 round loop，一般收尾看到旗標未設，便把零完成輪交給 QICK 平均，得到 NaN。

## 考慮過的選項與比較

- Stop 有三個選擇：強制 5 秒、強制 30 秒、或不訂上限而補檢查點。使用者選第三個，再加 5 秒放點目標。這遵循 [ADR-0066](../adr/0066-operation-lifecycle.md)：Cancel 是請求，終態和已完成結果由 domain owner 決定。
- 取消傳輸可延伸 SoC 的 request 生命週期，也可將旗標寫進 `AcquireRequest`、複製 QICK decimated round 後明傳，或保留 pacing-only。SoC 本來就借用 pacing 的查詢，讓同一查詢也進求解，不必更動 QICK 入口或數值 request。
- 零輪可在 mixin 直接拋既有例外，也可新增 backend 停止狀態，讓一般 acquisition 收尾判定；第三個選擇是要求 backend 只能在旗標為真時拋 `Cancelled`。採第一種，兩種模擬模式清理後一致處理，實機不變。

## 被否決的選項與理由

- 硬上限會變成延遲保證。30 秒是牆鐘工具的寬限，不是操作員 Stop 的目標。
- Pacing-only 無法中斷求解。複製 QICK round 會新增一份必須隨 QICK 升版對照的流程；把旗標放進 `AcquireRequest` 則把執行期控制混入數值 request。
- 為零輪多設停止狀態，需要另定傳遞與重置。要求旗標為真會收窄 backend 的契約。代替 caller 呼叫 `set()` 也不行：`AcquireCancelFlag.set()` 表示作者指定的資料停止條件已滿足，不代表 backend 中斷。

## 留下的問題

單點的 plan、WorkPlan 仍各有超過 5 秒的區段，只量到外層。首個 cold chunk 在空 cache、warmup 關閉下量到 5.4 至 8.5 秒；這不是 GUI 路徑的條件。單點在 600 秒逾時，Engine 之後的讀出分布、raw 準備和抽樣未覆蓋，也不外推 51 點首輪。後續 `ftf-stop-remaining-gaps` 先定位 plan 與 WorkPlan 子段，再量 GUI 條件下的 cold 路徑，並裁定如何覆蓋 Engine 後區段。

## 依據

原設定 51 gains、100 reps、100 rounds 的離線重測，在第 10 個 span 組裝時送 Stop。旗標 0.0009 秒後被讀到，0.017 秒到離線 core 終態，沒有完成輪。這個終點不含 Qt 與 Controller 收尾，不能與舊的 GUI 24.9 至 34.2 秒當成同端點比較。取消重測與單點診斷都是非排他。原始量測記錄未入庫，結論所需的秒數與條件留在本文。

真 QICK 0.2.436 的公開 acquisition 測試涵蓋兩種模擬模式。首輪在旗標仍清除時拋 `Cancelled`，修前沒有拋出停止例外；修後零輪拋例外，後輪中斷仍只平均完成輪，caller 旗標不變。Caller 可觀察的契約見 [simulation README](../../lib/zcu_tools/program/v2/simulation/README.md)。
