---
date: 2026-11-22
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - 2026-11-16-acquire-wall-analysis.md
  - 2026-11-16-shared-simulator-measurement.md
  - 2026-11-22-ftf-projection-weight-floor.md
basis:
  repo_commit: 7f43d78d6b4098655398c3f07d3dd234bc2e26f7
  external_sources: none
---

# 舊實測數字當現況

## 結論

2026-11-22，使用者選 B：「那采用建议吧，选 B。」使用者看過規則文字及 telemetry 收窄後答「可以。」規則寫入 [CLAUDE.md〈文檔組織與搜尋〉](../../CLAUDE.md#文檔組織與搜尋)，量測來源由呼叫端提供的邊界寫入 [ADR-0078〈模擬器帶通用調試介面〉](../adr/0078-evolution-module-boundary.md#模擬器帶通用調試介面)。

## 問題

1. `chain/freq` 的 materialize 在 `4a2dea8` 量得 325.150 s。這是 `ftf-demo/initial`、`time_scale=0`、3 workers、11 點和 187 輪的總和。派研究時沒帶量測 commit 和範圍，把數字當成現況瓶頸。兩份調查和一次小型 profile 後才確認 `b6bdb8024` 已加入 raw plan 重用。小型 profile 不是原場景的重測。
2. 同一批的 `chain/t1` 和 `chain/len_rabi` 在 Run 期限 360 s 逾時。兩列 `complete=false`，但 `materialize_wall_s=0.0` 被讀成「沒有 raw 工作」。兩個 reducer 都要求 raw；當時只是還沒走到該階段。舊 writer 用同一個零表示未到與量得零。
3. Heralded RB 的 74.3% 是 `666eb48`、Round 20 ready cfg、`low-snr.yaml`、seed 123、1 worker、`time_scale=0` 下，CPU DP5(4) 的 1,440 條雙 tone span 佔 20 格編譯加取數總耗時的比例。5–13 h 是同一來源外推至 8–16 seeds、depth 8 的完整 RB，不是實測。後續投資選項仍用了舊工作量，直到另一輪調查把外推標成失效。沒有新版完整 RB 的實測可用來取代它。
4. Projection floor 的升級清單沿用舊 history 模型下「第一點精度未驗證」的例外。Markov 路徑已取代舊模型，後續裁決已撤銷例外，升級清單卻沒跟上。任務退回重送。見 [FTF 樣板的 projective floor](2026-11-22-ftf-projection-weight-floor.md)。

引用數字時會丟掉來源與範圍。用數字派工前沒有比對量測後的程式改動，引用決策狀態前也沒有查取代紀錄。資料層另有一個缺陷：沒有走到的階段與量得零無法區分。

## 考慮過的選項與比較

- A：只擴充引用規則。數字旁留 commit、輸入與範圍，驅動決策前查被測路徑，引用裁決前查取代紀錄。
- B：採 A，並修正 telemetry。未完成列沒走到的階段寫 `null`，每列帶呼叫端提供的 commit 和 dirty。
- C：採 B，再定義固定的數字戳記格式與檢查腳本，由腳本列出量測後改到被測路徑的提交。
- D：由各讀取端對照目前的來源與輸入，拒讀過時資料。這能攔下直接使用過時資料的讀取者，但每個讀取端要先定義比對範圍。

## 被否決的選項與理由

A 修不了資料層的歧義。B 保留 [Acquire wall 分組](2026-11-16-acquire-wall-analysis.md) 對正常返回列的做法，但把失敗尚未到達的階段與不需要的階段分開。它也遵循 [共用計時](2026-11-16-shared-simulator-measurement.md) 的先例，實測零原樣保留。決策前查版本，再決定是否重測，沿用 [ADR-0078 的量測分層](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)，不拿小型 profile 代替整列實驗。

C 留到再發生同類事件時。先讓引用規則與資料修正各處理已知問題，不先訂一套通用戳記。D 的比對成本最高：例如 Run 牆鐘續跑核對 commit 和 fixture 檔案，family 量表核對場景 fingerprint，兩者不能共用「過時」的判斷。

## 留下的問題

歷史 telemetry 的零無法從欄位本身判斷是否量過。本次不回填歷史資料，也不新增 freshness gate。若又用舊數字驅動派工、裁決或驗收，再升到 C。

## 依據

原始量測收據、事件調查及使用者問答未入庫。上文保留裁決所用的 commit、設定、範圍與可確認的後果。沒有收據的人工成本不換算為耗時。既有量測欄位以 [共用 acquire telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 和 [Run 牆鐘工具](../../tools/README.md#實驗-run-牆鐘) 為準。
