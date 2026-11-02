# 討論紀錄索引

**讀者：** framework-dev、agent。透過本索引與各篇討論快照查找設計討論、選項與裁決脈絡。

討論紀錄是當時的快照，不擁有決策。撰寫規則見 [CLAUDE.md](../../CLAUDE.md) 的「討論紀錄」。

## Agent 介面

- 2026-10-12 [QCA 作為前端的可行性評估](2026-10-12-qca-frontend-evaluation.md)：QCA 不取代 measure-gui MCP，只借鏡規劃判準與校準 VLM 判圖。
- 2026-10-12 [Agent 介面分層的推導](2026-10-12-agent-interface-layers.md)：自動化往下推，agent 只在宣告的判斷交接處介入，決策見 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 2026-10-12 [fluxdep MCP 的定位與介面重設計](2026-10-12-fluxdep-mcp-design.md)：依 agent 決策點組裝 RPC 與證據圖，介面原則見 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 2026-10-16 [Recipe 與 workflow 的協作模型：generator、阻塞函式或 async](2026-10-16-recipe-workflow-yield-vs-async.md)：Generator 的歷史選項已由 [ADR-0062](../adr/0062-experiment-workflow.md) 與 [ADR-0075](../adr/0075-agent-interface-layers.md) 的 async 契約取代。
- 2026-10-21 [Recipe 接口重設計](2026-10-21-recipe-interface-redesign.md)：記錄 Wait／Ask 引擎與有型別 recipe 入口的歷史設計，現行作者介面見 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 2026-10-21 [三角色協定：Wait／Ask／Signal](2026-10-21-three-role-protocol.md)：Wait／Ask／Signal 的 generator 協定已由 [ADR-0062](../adr/0062-experiment-workflow.md) 與 [ADR-0075](../adr/0075-agent-interface-layers.md) 取代。
- 2026-10-21 [Workflow 定位、承載與執行基底](2026-10-21-workflow-positioning-and-hosting.md)：Recipe 操作 GUI tab，workflow 跑實驗核心並由 measure-gui 承載，現行分工見 [ADR-0062](../adr/0062-experiment-workflow.md) 與 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 2026-10-22 [Workflow 改為單一 generator：取消、Signal、step 模型與回退](2026-10-22-workflow-single-generator.md)：單一 generator 與 Pause／Resume 的歷史設計已由 [ADR-0062](../adr/0062-experiment-workflow.md) 與 [ADR-0075](../adr/0075-agent-interface-layers.md) 取代。
- 2026-10-22 [中斷 scope 成為顯式的 core 契約](2026-10-22-explicit-scope-contract.md)：中斷由顯式 Scope 區分目標並保證作用域，決策見 [ADR-0075](../adr/0075-agent-interface-layers.md)。

## 模擬量測環境

- 2026-10-13 [模擬量測環境、低能投影與 FTF preset 的討論](2026-10-13-simulation-environment-ftf-preset.md)：保留環境與 preset 的早期選項，後續方向見[模擬環境草稿](../adr/draft/simulation-environments.md)。
- 2026-10-15 [離線GPU建庫、分段演化與插值重用](2026-10-15-offline-propagator-library.md)：離線建庫的早期方案已由 [FTF 模擬加速討論](2026-10-16-ftf-simulation-speedup.md) 的 runtime 路線取代。
- 2026-10-16 [FTF 模擬加速：快速路徑、shot 統計、分段快取與分派決策圖](2026-10-16-ftf-simulation-speedup.md)：每點演化一次後做 shot 統計抽樣，以 runtime 分派和快取取代離線 U 表與 density record。
- 2026-10-17 [FTF 分段引擎：Dyson 門檻、分類成本與事後檢查](2026-10-17-ftf-dyson-guard-and-classification.md)：分類只做輕量判斷，Dyson 的 apply 後偏差只報告不重算，同一段不演化兩次。
- 2026-10-18 [Dyson 的解析分類、輸入共用與 sparse scan](2026-10-18-dyson-input-reuse.md)：解析 B2 與 post-check 分工並共用前綴輸入，sparse scan 僅省 12.6% 因而未採用。
- 2026-10-18 [Constant-tone period builders 的演進](2026-10-18-constant-period-builders.md)：比較 CF4、Padé7、Sambe 與 gain anchors，保留 carrier continuation 零命中及 RSS 超標的否決紀錄。
- 2026-10-18 [Shaped periods、gate preparation 與 carrier reuse](2026-10-18-shaped-period-and-gates.md)：採 build-once handoff、phase-adaptive gates 與 narrow carrier interpolation，wide scan 保留 exact bypass。
- 2026-10-18 [Two-tone Floquet 的 frame、sidebands 與成本交叉](2026-10-18-two-tone-floquet.md)：保留 divisor 修正、P residual 失敗與未認證假設，短段 throughput 改走 direct。
- 2026-10-18 [Direct integration 的 RHS、step controls 與 interaction picture](2026-10-18-direct-integration.md)：採 Hermitian RHS、sample-spacing ceiling 與 row-independent IP，撤掉依賴 partition 的 gain grouping。
- 2026-10-18 [Readout 的 guarded discrete sums](2026-10-18-readout-discrete-sums.md)：Flat integrated readout 直接求 ADC 樣本和，unsupported contents 保留 field fallback。
- 2026-10-18 [Planner 的 evidence admission 與固定成本](2026-10-18-planner-evidence-and-overhead.md)：Accuracy 通過但 144 組比較有 100 組較慢，未取得 performance admission。
- 2026-10-21 [CKP reduced closure 研究：精度過但成本 42 倍](2026-10-21-ckp-reduced-closure-study.md)：Displaced-Fock N2 精度過但成本為半古典的 41.9 倍，否決 Gaussian 並轉向 joint Fock 性能研究。
- 2026-10-21 [Joint Fock CKP 性能研究：保守 RK4 加 N+1 cutoff 外推 129 s](2026-10-21-joint-fock-performance-study.md)：兩能階保守版外推 129 s，16 能階成本不可行，改評估動態 [0,2] 與四階診斷。
- 2026-10-28 [FTF 固定 DC flux ramp 的截斷與成本界限](2026-10-28-ftf-fixed-dc-flux-benchmark.md)：M16 僅為最大成功尺寸，M24／M32 reference 失敗，不能認證大幅 excursion 或 reset fidelity。
- 2026-10-29 [Module 改 QICK Macro 與模擬 frontend 改讀 macro](2026-10-29-module-to-macro.md)：記錄 Module 改 macro 與具名 timeline 的原型比較及六次後記，現行決策見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。

## Program 與 timeline 編譯

- 2026-10-31 [Timeline 編譯：變數模型、通用 join 與 s14 事件時間](2026-10-31-timeline-variable-model.md)：Timeline 合流採共用 join，範圍證明改用 Abs 表示，現行決策見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。

## Cfg 與表單

- 2026-10-13 [Measure 表單重設計：從 builder 重抄到 core cfg 推導](2026-10-13-measure-form-redesign.md)：表單形狀從 core cfg 推導，說明與初值各有 owner，決策見 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-13 [表單選項的分層：生成器給選項，widget 照清單畫](2026-10-13-form-options-layering.md)：生成器給選項而 widget 只呈現，現行分工見 [ADR-0065](../adr/0065-cfg-editing.md) 與 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-17 [Measure Form 的校準初值與 sweep 模式切換](2026-10-17-measure-form-initial-values-and-sweep-mode.md)：校準初值在開 Form 時算成數字，缺值、無效值與故障分流，決策見 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-29 [Singleshot 分類參數放進 readout cfg](2026-10-29-readout-classifier.md)：Classifier 隨 readout cfg 保存，作者明確傳入 acquisition，決策見 [ADR-0074](../adr/0074-flat-readout-module.md)。

## 模組邊界

- 2026-10-20 [框架去具體化：alias、模板、predictor 與一晶片一 qubit 假設](2026-10-20-framework-decoupling.md)：領域 alias、模板與模型認知留在 zcu_lab，框架的 owner 邊界見 [ADR-0072](../adr/0072-module-boundaries.md)。

## 測試與驗證

- 2026-10-16 [Agent 自行做 UI e2e：元件驅動情境](2026-10-16-agent-ui-e2e.md)：Agent 經使用者可操作的元件驗證可見行為，使用者走查判品味。

## 存檔與資料格式

- 2026-10-14 [量測成果 Artifact、設定快照與讀回的推導](2026-10-14-artifact-persistence.md)：設定快照正本只在 data.h5，保存與讀回決策見 [ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0069](../adr/0069-experiment-core-and-frontends.md) 與 [ADR-0067](../adr/0067-gui-application.md)。
- 2026-10-23 [Store 讀取效率與寫入慣例](2026-10-23-store-read-efficiency.md)：保留分組提交，以增量讀和內容 hash 快照減少重複驗證。

## 量測策略

- 2026-10-27 [T1/T2 over flux 量測策略：哪些機制有效](2026-10-27-coherence-strategy-levers.md)：自適應 flux 選點無增益，穩健 delay 取樣讓每條曲線的 σ 約為原來 0.73 倍。
- 2026-10-29 [Heralding：取數層 post-selection，不做 FPGA active reset](2026-10-29-heralding-post-selection.md)：在 host 取數層依 herald 篩選再平均，不做 FPGA active reset，決策見 [ADR-0074](../adr/0074-flat-readout-module.md)。
- 2026-10-30 [Readout 優化指標：用兩次讀出的一、二次矩算 SNR 與 QND](2026-10-30-readout-optimization-metrics.md)：兩次讀出的一、二次矩估 SNR 與 QND，先畫 Pareto 前緣由人選。
- 2026-11-02 [Heralding：herald／signal 二階矩自校準的轉移矩陣與指標 C](2026-11-02-herald-transfer-matrix.md)：IQ 改二階特徵，從 herald 與 signal 的 36 個總和自校準，只用 C = 1 − tr J 分析，不需要 classifier。
- 2026-11-03 [重複（A + readout）讀出鏈：效率、讀出 MIST 與三視角分解](2026-11-03-repeated-readout-chain.md)：重複讀出讓精度約隨 √N 提高，讀出 MIST 沿鏈累積限制 N；三視角分解遇讀出躍遷時沒有優勢。
