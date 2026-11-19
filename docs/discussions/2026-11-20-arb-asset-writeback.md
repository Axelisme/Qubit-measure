---
date: 2026-11-20
status: active
superseded_by: null
related:
  - ../adr/0008-cfg-editor-session.md
  - ../adr/0063-persistence-ownership.md
  - 2026-11-20-in-memory-arb-waveforms.md
  - ../../lib/zcu_tools/resources/waveform_assets.md
basis:
  repo: a987118b6
  external: none
---

# 用 writeback 發布 CZ 的 arb 資產

## 結論

Analyze 提出 `ArbAssetWriteback`，不寫檔。使用者可在 GUI 或 MCP 接受發布與引用新 key 的 module 提案。CZ 優化的例子是讓 `cz_f1` 和 `cz_f2` 指向同一份最佳樣本，再更新 local Z。資產屬於 entry，該 entry 的 global 與各 point 共用。Executor 先提交資產，再提交 parameters、library 與 records。發布不覆寫舊資產。現行規則見 [ADR-0008](../adr/0008-cfg-editor-session.md#writeback-proposal-與提交) 與 [ADR-0063](../adr/0063-persistence-ownership.md#失敗路徑與遷移邊界)。

## 問題

當時的 module writeback 只能寫 library 參考，不能發布它指向的 NPZ。GUI 的 arb 編輯對話框只編公式，MCP 沒有發布原始樣本的入口。若 Analyze 先把最佳樣本寫進庫，使用者即使不接受引用，庫裡還是多一個檔案。

## 考慮過的選項與比較

- 新增發布提案。GUI 和 MCP 都能預覽並選擇接受。資產排第一，library 開始寫時就已有新 key。
- 讓 Analyze 直接寫 NPZ，writeback 只改引用。這樣不必擴充 Apply，但無法用 Apply 決定要不要保存樣本。
- 讓 recipe 寫檔。使用者在 2026-11-19 補充：「然后发布 arbitrary 资产这是可以的，但是我其实觉得 recipe 内想要写档案也是没有问题的。」比較過 writeback 可供 GUI 與 MCP 使用、recipe 寫檔只有 MCP 途徑後，使用者同日對維持 writeback 案答：「可以」。

這次遵循 [ADR-0008](../adr/0008-cfg-editor-session.md) 的提案與提交分離，以及 [ADR-0063](../adr/0063-persistence-ownership.md) 的 accepted ledger 和分組提交。與 [range_find 的參數寫回](2026-11-16-range-find-writeback-upsert.md) 不同，既有同名資產可能已有別的 module 引用。相同的 time、I、Q 算已存在，原檔不動。樣本不同則整個資產組不寫，後續組也不提交。

以下兩項是 2026-11-20 agent 代裁，待使用者覆核：

- Apply 檢查已選 module 能沿作用中 library 解析出的 arb key。Key 必須已在作用中 entry，或有同份 draft 的已選資產提案。找不到資產時，在寫 accepted 前報出提案與 key。找不到 library child 時沿用原行為，不在這一步新增錯誤。Primary 與 Post 分開提交，後提交的 draft 不能替先提交的 draft 補 key。
- CZ 優化用 `cz_opt_` 加樣本雜湊的前 12 個十六進位字元當 key。重提同一組樣本時可沿用先前已建的資產。即使短雜湊撞 key，executor 也會比對完整樣本，不會覆寫。

## 被否決的選項與理由

- Analyze 直接寫檔會在接受前改動資產庫，使用者不採用結果時仍留下檔案。
- Recipe 直接寫檔不作為此實驗的發布路徑，因為 GUI 沒有同一條操作路徑。這不是禁止 recipe 寫檔的一般規則。

## 留下的問題

成功發布資產不代表整批寫回成功。若 library 寫入失敗，executor 不移除已發布的資產。可重新接受同一份樣本，資產組會把同 key、同內容回報為已存在。

使用者在 2026-11-19 表明「我希望最后知识库能整理出一个流程决策树给我审阅」，並提醒實機 single-shot 品質可能遠低於模擬。低 SNR 下的量測選法仍需用實機或低 SNR 模擬的結果驗證。

## 依據

當時的設計比較與原始工作檔未入庫。實作可核對 [resources executor](../../lib/zcu_tools/resources/entry/writeback.py)、[資產發布規則](../../lib/zcu_tools/resources/waveform_assets.md) 與 [Measure 的 draft 與 Apply](../../lib/zcu_tools/gui/app/measure/writeback/README.md)。

代裁依使用者 2026-11-20 的授權：「STDO:我去睡了，我看過目前規劃方向沒有問題，請你之後幫我代裁設計與推進任務，都不需要問我，你自己決定」。
