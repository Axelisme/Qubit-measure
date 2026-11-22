---
date: 2026-11-23
status: active
superseded_by: null
related:
  - 2026-11-22-wall-time-fixture-flux-routing.md
  - 2026-11-21-wall-time-coverage.md
  - 2026-11-21-simulator-core-time.md
  - ../adr/0072-module-boundaries.md
  - ../adr/0078-evolution-module-boundary.md
basis:
  repo_commit: 3e12ee9dc3da4e7486502d5e63ad6551e44074c6
  standard_preset_commit: b1308a924ff9b650abb9837c55aaac52b8b41298
  initial_workload_selection_commit: 4a2dea87690bfcb2764886b2e5feb0b2113510c9
  external_sources: none
---

# 牆鐘工具量手選場景

## 結論

[牆鐘工具](../../tools/README.md#實驗-run-牆鐘)自備 FTF 設定、乾淨工作點和場景清單，只透過 `zcu_lab.assembly` 取得元件與實驗的宣告。每個場景開現行 Form，寫入有界覆寫，再走原 Controller Run。Lab 更改標準 preset 或新增 adapter 不會改量表。Lab 更改已選的 Form，下次就用新版。GUI 與 MCP 繼續使用各自原有的標準 preset、示範 point 和 Form 初值。

首批選入 52 個已有選入理由的工作量，以及原先缺快磁通或共用 coil 的 10 個派送驗收場景。Fixture 從已驗的設定和 point 資料一次建立，不繼承舊 classifier 或 Run ledger。十列從交付版公開入口驗 Form Ready 與帶 admission token 的 RunStarted，不要求慢列完成。這不替代另行快磁通驗收，也不改正式全表的執行時點。

## 問題

舊全表隨 live adapter inventory 增長，讓新增實驗改變統計分母。共用標準 preset 的候選還會在 lab 調整雜訊或換工作點時改變計時場景，即使維護者沒有改過量測輸入。直接讀 lab 內部模擬檔與 startup 也不符合 [ADR-0072](../adr/0072-module-boundaries.md#使用者空間的依賴原則)。

舊報表還用 adapter 聲明的路徑掃描整庫 guide。這可回答文件覆蓋問題，但不驗證一個所選場景能否派送。使用者選擇讓量測工具只檢查會影響自己輸出的事項。

## 考慮過的選項與比較

- 使用者先確認工具應自備一份資料。這增加一份設定，換來維護者可自行決定何時改工作量。工具只在初始化時取用已有依據的標準值，之後不跟 lab 同步。前一份缺路由的 fixture 不能直接復用。
- 自持場景，但用 live adapter 清單選列，仍會讓新增 adapter 改分母。改用人工選入、不依賴 live inventory 的 manifest。每列有穩定 ID、精確 adapter key 和選入理由；同一 adapter 可量多種工作量。
- 工具跑現行實驗。當時比較了複製核心演化序列與呼叫實驗定義；使用者以維護為先，選擇後者。[ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e) 保留這支完整 Run 工具的用途，固定 optimizer benchmark 仍各自持有序列並比較 `Engine.evolve`。

## 被否決的選項與理由

- 跟隨 GUI/MCP preset，或替工具建立自動同步。兩者都讓例行 lab 變更改寫原排定的輸入。
- 搬入自己的演化序列。工具便不能告訴維護者現行實驗的耗時，還要維護另一份序列。
- 繼續在牆鐘表內做整庫 guide 稽核。用不到 guide 的場景也能計時；工具只需核對已選場景引用的 adapter。移除稽核不改 adapter 的 `guide_path` 或 GUI guide。

## 留下的問題

若 adapter 失效，維護者要更新或刪除相應場景。報表保留未量原因，不造 Run 時間、不重用該列。180 秒判定只涵蓋所選場景。需要有序 Local Z 配對或 GE Post classifier 的場景仍須滿足各自前置；缺前置不等於模擬器不支援。舊批次應回原 commit 重建；新批次只有在相同 commit 與輸入雜湊時才可續跑。新舊 projection floor 不同，不能把跨 fixture 的耗時差直接當成優化收益。

## 依據

- 2026-11-22，使用者：「沒有，我還是覺得工具應該自備一份」。2026-11-23 補充：「它其实不是量现在实验室有每个实验的时间，它是量特挑的场景模拟的时间。」比較演化序列與實驗定義後答：「以好维护为优先。」「就照你的建议，以呼叫实验定义来聊就好了。」
- 2026-11-23，對移除 guide 稽核，使用者：「A，不要做多餘的事，想檢查格式是否合理應該是其他工具負責，不是計時工具，類似的工具都該遵守KISS原則」。檢查 scene 的精確 adapter 是否存在仍屬於計時工具的責任。
- [入庫 manifest](../../tools/data/experiment_wall_time/case-inputs.json) 保存首批場景的理由。早期的場景調查、選集工作檔及原始對話未入庫；兩次工作的實測收據不能互代。
