# 討論紀錄索引

討論紀錄是當時的快照，不擁有決策。撰寫規則見 [CLAUDE.md](../../CLAUDE.md) 的「討論紀錄」。

## Agent 介面

- 2026-10-12 [QCA 作為前端的可行性評估](2026-10-12-qca-frontend-evaluation.md)：QCA 的儀器層較弱，不取代 measure-gui MCP；只借鏡規劃時約定判準與校準 VLM 判圖。
- 2026-10-12 [Agent 介面分層的推導](2026-10-12-agent-interface-layers.md)：自動化往下推，agent 只在宣告的判斷交接處介入；決策見 ADR-0075 Agent 介面分層。
- 2026-10-12 [fluxdep MCP 的定位與介面重設計](2026-10-12-fluxdep-mcp-design.md)：MCP 依 agent 的六個決策點與能力清單組裝 RPC，每張譜各自保有 context，證據圖只在決策點回；並歸納出十條適用於全部 MCP 的設計範式。
- 2026-10-16 [Recipe 與 workflow 的協作模型：generator、阻塞函式或 async](2026-10-16-recipe-workflow-yield-vs-async.md)：保留 generator 由 driver 掌控進度，recipe 改用有型別的 `yield from` helper；主動訊息只支援控制命令與邊界生效的 tunables，不做 async。

## 模擬量測環境

- 2026-10-15 [離線GPU建庫、分段演化與插值重用](2026-10-15-offline-propagator-library.md)：固定模型採離線建庫存檔、模擬載入插值；CPU可行性已有證據，GPU建表與檔案載入仍待驗證，四階精度以量測用途評估。
- 2026-10-13 [模擬量測環境、低能投影與 FTF preset 的討論](2026-10-13-simulation-environment-ftf-preset.md)：早期選項與 prototype 的歷史快照；後續決策以[模擬環境 ADR](../adr/draft/simulation-environments.md)為準，文末後記說明已定案與仍待完成的範圍。

## Cfg 與表單

- 2026-10-13 [Measure 表單重設計：從 builder 重抄到 core cfg 推導](2026-10-13-measure-form-redesign.md)：表單形狀從 core cfg 推導、說明放 core、起始值寫在 form.py 樹，角色表退化成範本目錄，`dev` 改具名欄位；決策見 ADR draft `measure-form-derivation`。
- 2026-10-13 [表單選項的分層：生成器給選項，widget 照清單畫](2026-10-13-form-options-layering.md)：選項清單由生成器產生、核心通用、widget 只照清單畫；換分支不繼承舊值；ref 轉 inline 用選項指令；key-or-inline 統一成有 tag 的形狀；決策見 Pydantic 表單 ADR draft。

## 測試與驗證

- 2026-10-16 [Agent 自行做 UI e2e：元件驅動情境](2026-10-16-agent-ui-e2e.md)：行為由 agent 用只經使用者可碰元件操作、只讀使用者可見狀態的情境驗證，使用者走查只判品味；框架情境進 `tests/`，實驗走查腳本留 task scratch。

## 存檔與資料格式

- 2026-10-14 [量測成果 Artifact、設定快照與讀回的推導](2026-10-14-artifact-persistence.md)：設定快照正本只在 `data.h5`，Notebook 與 GUI 都讀它；Artifact 以項目種類擴充、格式由框架決定，讀回採資料優先、設定可缺；決策見 ADR-0063／0069／0067。
