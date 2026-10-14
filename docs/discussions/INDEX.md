# 討論紀錄索引

討論紀錄是當時的快照，不擁有決策。撰寫規則見 [CLAUDE.md](../../CLAUDE.md) 的「討論紀錄」。

## Agent 介面

- 2026-10-12 [QCA 作為前端的可行性評估](2026-10-12-qca-frontend-evaluation.md)：QCA 的儀器層較弱，不取代 measure-gui MCP；只借鏡規劃時約定判準與校準 VLM 判圖。
- 2026-10-12 [Agent 介面分層的推導](2026-10-12-agent-interface-layers.md)：自動化往下推，agent 只在宣告的判斷交接處介入；決策見 Agent 介面分層 ADR draft。
- 2026-10-12 [fluxdep MCP 的定位與介面重設計](2026-10-12-fluxdep-mcp-design.md)：MCP 依 agent 的六個決策點與能力清單組裝 RPC，每張譜各自保有 context，證據圖只在決策點回；並歸納出十條適用於全部 MCP 的設計範式。

## Cfg 與表單

- 2026-10-13 [Measure 表單重設計：從 builder 重抄到 core cfg 推導](2026-10-13-measure-form-redesign.md)：表單形狀從 core cfg 推導、說明放 core、起始值寫在 form.py 樹，角色表退化成範本目錄，`dev` 改具名欄位；決策見 ADR draft `measure-form-derivation`。

## 存檔與資料格式

- 2026-10-14 [量測成果 Artifact、設定快照與讀回的推導](2026-10-14-artifact-persistence.md)：設定快照正本只在 `data.h5`，Notebook 與 GUI 都讀它；Artifact 以項目種類擴充、格式由框架決定，讀回採資料優先、設定可缺；決策見 ADR draft `artifact-persistence`。
