# 討論紀錄索引

**讀者：** framework-dev、agent。透過本索引與各篇討論快照查找設計討論、選項與裁決脈絡。

討論紀錄是當時的快照，不擁有決策。撰寫規則見 [CLAUDE.md](../../CLAUDE.md) 的「討論紀錄」。

## Agent 介面

- 2026-10-12 [QCA 作為前端的可行性評估](2026-10-12-qca-frontend-evaluation.md)：QCA 的儀器層較弱，不取代 measure-gui MCP；只借鏡規劃時約定判準與校準 VLM 判圖。
- 2026-10-12 [Agent 介面分層的推導](2026-10-12-agent-interface-layers.md)：自動化往下推，agent 只在宣告的判斷交接處介入；決策見 [ADR-0075 Agent 介面分層](../adr/0075-agent-interface-layers.md)。
- 2026-10-12 [fluxdep MCP 的定位與介面重設計](2026-10-12-fluxdep-mcp-design.md)：MCP 依 agent 的六個決策點與能力清單組裝 RPC，每張譜各自保有 context，證據圖只在決策點回；並歸納出十條適用於全部 MCP 的設計範式。
- 2026-10-16 [Recipe 與 workflow 的協作模型：generator、阻塞函式或 async](2026-10-16-recipe-workflow-yield-vs-async.md)：保留 generator 由 driver 掌控進度，recipe 改用有型別的 `yield from` helper；主動訊息只支援控制命令與邊界生效的 tunables，不做 async。
- 2026-10-21 [Recipe 接口重設計](2026-10-21-recipe-interface-redesign.md)：兩層執行（驅動引擎只懂 Wait／Ask／return，便利層包 GUI RPC）；輸入是作者定義的 FormModel、原生型別、兩態 None 保留；`@recipe` 取代 DEFINITION；回覆為作者 report 的包裝。
- 2026-10-21 [三角色協定：Wait／Ask／Signal](2026-10-21-three-role-protocol.md)：agent、腳本、runtime 三角色；recipe 與 workflow Engine 底層同一份協定，effect 與 env 方法都是糖；Signal 第一版只有內建 kind 由 host 處理。
- 2026-10-21 [Workflow 定位、承載與執行基底](2026-10-21-workflow-positioning-and-hosting.md)：recipe 操作 GUI tab 可多實驗，workflow 跑 Engine；workflow 由 measure-gui 承載；維持 helper，helper 搬到實驗目錄並列 core，共用零件不共用流程。
- 2026-10-22 [Workflow 改為單一 generator：取消、Signal、step 模型與回退](2026-10-22-workflow-single-generator.md)：recipe 取消統一 return；協定移除 Signal；workflow 由 step 函式改為單一 generator 並刪 Effect，Pause 立即打斷並於 Resume 重跑 operation；回退走新 run 組合，engine 不提供整點重做。
- 2026-10-22 [中斷 scope 成為顯式的 core 契約](2026-10-22-explicit-scope-contract.md)：協作式與錯誤式中斷依「我等的事」或「我」被中斷分流；Scope 由 experiment 層定義、core 與 workflow 共用，`with`／`for` 保證邊界，路徑 `kind:name#seq`，目錄鏡射 scope；決策見 [ADR-0075](../adr/0075-agent-interface-layers.md)。

## 模擬量測環境

- 2026-10-21 [CKP reduced closure 研究：精度過但成本 42 倍](2026-10-21-ckp-reduced-closure-study.md)：flip 非真空腔激發必須保留；displaced-Fock N2 精度過但成本 41.9 倍，Gaussian 否決；半古典先 partial land，下一步走 joint Fock 性能研究。
- 2026-10-21 [Joint Fock CKP 性能研究：保守 RK4 加 N+1 cutoff 外推 129 s](2026-10-21-joint-fock-performance-study.md)：兩能階候選通過局部精度並外推 129 s，快版餘裕薄不採；16 能階原型數十小時不可行，改動態 [0,2] 加四階診斷；下一步保守版 prototype 與全圖驗證。
- 2026-10-18 [Dyson 的解析分類、輸入共用與 sparse scan](2026-10-18-dyson-input-reuse.md)：解析 B2 與 post-check 分工，共用前綴收縮輸入，sparse scan 只省 12.6% 因而未採用。
- 2026-10-18 [Constant-tone period builders 的演進](2026-10-18-constant-period-builders.md)：比較 CF4、Padé7、Sambe 與 gain anchors，保留 carrier continuation 零命中及 RSS 超標的否決紀錄。
- 2026-10-18 [Shaped periods、gate preparation 與 carrier reuse](2026-10-18-shaped-period-and-gates.md)：採 build-once handoff、phase-adaptive gates 與 narrow carrier interpolation，wide scan 保留 exact bypass。
- 2026-10-18 [Two-tone Floquet 的 frame、sidebands 與成本交叉](2026-10-18-two-tone-floquet.md)：保留 divisor 修正、P residual 失敗與四項未認證假設，短段 throughput 最後改走 direct。
- 2026-10-18 [Direct integration 的 RHS、step controls 與 interaction picture](2026-10-18-direct-integration.md)：採 Hermitian RHS、sample-spacing ceiling 與 row-independent IP，撤掉依賴 partition 的 gain grouping。
- 2026-10-18 [Readout 的 guarded discrete sums](2026-10-18-readout-discrete-sums.md)：flat integrated readout 直接求 ADC 樣本和，unsupported contents 保留 field fallback，正式比較另補 clean pairs。
- 2026-10-18 [Planner 的 evidence admission 與固定成本](2026-10-18-planner-evidence-and-overhead.md)：accuracy 通過仍未取得 performance admission，144 組比較有 100 組較慢，stage estimates 不代替完整 acquire。

- 2026-10-17 [FTF 分段引擎：Dyson 門檻、分類成本與事後檢查](2026-10-17-ftf-dyson-guard-and-classification.md)：分類只做輕量判斷，同一段不演化兩次；Dyson 的 |Δp| 改為 apply 後只報告不重算，治本靠分類時的解析指標；門檻由每段預算推出，係數由掃描校準；不用 Rust。
- 2026-10-16 [FTF 模擬加速：快速路徑、shot 統計、分段快取與分派決策圖](2026-10-16-ftf-simulation-speedup.md)：主要瓶頸是 acquire 重複 reps × rounds 次。改為每點只算一次並用統計抽樣，各段依決策圖分派到 Dyson、週期、靜態、gate 快取、Floquet 或 numba DOP853，使用者規模從約 8 天降到 13 分鐘至 2 小時；只用 numba，刪除離線 U 表與 density record。
- 2026-10-15 [離線GPU建庫、分段演化與插值重用](2026-10-15-offline-propagator-library.md)：固定模型採離線建庫存檔、模擬載入插值；CPU可行性已有證據，GPU建表與檔案載入仍待驗證，四階精度以量測用途評估。已被 2026-10-16 的 FTF 模擬加速取代：FTF 一定有耗散，離線 U 表會刪除，改用 runtime lazy 快取。
- 2026-10-13 [模擬量測環境、低能投影與 FTF preset 的討論](2026-10-13-simulation-environment-ftf-preset.md)：早期選項與 prototype 的歷史快照；後續決策以[模擬環境 ADR](../adr/draft/simulation-environments.md)為準，文末後記說明已定案與仍待完成的範圍。

## Cfg 與表單

- 2026-10-17 [Measure Form 的校準初值與 sweep 模式切換](2026-10-17-measure-form-initial-values-and-sweep-mode.md)：缺值給公允值、無效值留空、故障報錯，「可預測與一致比正確重要」，初值開 Form 時算成數字；sweep 模式存於共用 Form 輸入，recipe 一次設定並重設為 range + expts。
- 2026-10-13 [Measure 表單重設計：從 builder 重抄到 core cfg 推導](2026-10-13-measure-form-redesign.md)：表單形狀從 core cfg 推導、說明放 core、起始值寫在 form.py 樹，角色表退化成範本目錄，`dev` 改具名欄位；決策見 [ADR-0073 Measure 表單推導](../adr/0073-measure-form-derivation.md)。
- 2026-10-13 [表單選項的分層：生成器給選項，widget 照清單畫](2026-10-13-form-options-layering.md)：選項清單由生成器產生、核心通用、widget 只照清單畫；換分支不繼承舊值；ref 轉 inline 用選項指令；key-or-inline 統一成有 tag 的形狀；歷史草稿為 Pydantic 表單 ADR，現行決策見 [ADR-0065](../adr/0065-cfg-editing.md) 與 [ADR-0073](../adr/0073-measure-form-derivation.md)。

## 模組邊界

- 2026-10-20 [框架去具體化：alias、模板、predictor 與一晶片一 qubit 假設](2026-10-20-framework-decoupling.md)：具體 alias、儀器與模型的認知放 zcu_lab；模板綁定只由呼叫端明填，alias 是 setup／point 各自的檔案；predictor 讀自己的格式檔；框架內建支援能力用內聚子模組而非注入；session 清掉 chip／qubit 假設，分析工具改 entry＋component。

## 測試與驗證

- 2026-10-16 [Agent 自行做 UI e2e：元件驅動情境](2026-10-16-agent-ui-e2e.md)：行為由 agent 用只經使用者可碰元件操作、只讀使用者可見狀態的情境驗證，使用者走查只判品味；框架情境進 `tests/`，實驗走查腳本留 task scratch。

## 存檔與資料格式

- 2026-10-23 [Store 讀取效率與寫入慣例](2026-10-23-store-read-efficiency.md)：保留分組提交，以 ledger 增量讀與內容 hash 快照減少重複驗證；批次為原形、工作量測試與 owner handler 延遲 warning 分別保護交易邊界、工作量與現場診斷。
- 2026-10-14 [量測成果 Artifact、設定快照與讀回的推導](2026-10-14-artifact-persistence.md)：設定快照正本只在 `data.h5`，Notebook 與 GUI 都讀它；Artifact 以項目種類擴充、格式由框架決定，讀回採資料優先、設定可缺；決策見 [ADR-0063](../adr/0063-persistence-ownership.md)／[0069](../adr/0069-experiment-core-and-frontends.md)／[0067](../adr/0067-gui-application.md)。

## 量測策略

- 2026-10-27 [T1/T2 over flux 量測策略：哪些機制有效](2026-10-27-coherence-strategy-levers.md)：自適應 flux 選點沒有增益；唯一有效的是每條曲線的穩健 delay 取樣（σ 約 0.73 倍）；不需新 workflow，建議在 `coherence_fluxdep` 加 delay 模式。
