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
- 2026-11-19 [Recipe 的寫回 helper 與 ask 原語](2026-11-19-recipe-ask-helpers.md)：原語與 helper。通用契約只有 ask 與正常結束；寫回 helper 依 stage 出題，原則見 [ADR-0075](../adr/0075-agent-interface-layers.md#通用契約只有-ask-與正常結束)。
- 2026-11-20 [Agent 只用 logical cfg 路徑；recipe overrides 支援 value、select、command](2026-11-20-logical-cfg-paths-and-overrides.md)：介面收窄。Agent 只看 logical 路徑，canonical 方法改 internal；overrides 由各 recipe 宣告，決策見 [ADR-0068](../adr/0068-remote-transport.md) 與 [ADR-0075](../adr/0075-agent-interface-layers.md)。
- 2026-11-20 [Recipe 報告與執行進度](2026-11-20-recipe-report-polish.md)：介面回報。轉出 owner 的 fit quality 與 STIRAP 網格峰值；取數中的 execution 顯示 Run 進度。
- 2026-11-20 [Two-qubit MCP recipes 的首版範圍與輸入](2026-11-20-two-qubit-recipes.md)：介面設計。首版選七個實驗，每個 recipe 跑一輪，由各 recipe 宣告 Open／Reset 欄位。
- 2026-11-20 [fluxdep GUI 跟隨 agent 操作](2026-11-20-fluxdep-gui-follows-agent.md)：顯示跟隨。成功的非唯讀操作切到目標譜、主頁與階段，不轉移控制權。

## 模擬量測環境

- 2026-11-24 [ADC raw 停線只放行一次 flat 跳轉試驗](2026-11-24-adc-raw-flat-jump-exception.md)：停線例外。只試跳轉所用的兩態基線，通過兩場配對後保留；其餘 raw 與統計量停線不變（designer 代裁，待使用者覆核）。
- 2026-11-23 [同頻多線在 Evolution 的表示](2026-11-23-same-carrier-merge.md)：模擬器路由。保留原 slots 的對照，同載波同形線摺成單 tone，異形線保留多 tone 路由。
- 2026-11-23 [模擬器優化嘗試記在哪裡](2026-11-23-simulator-attempts-log.md)：文件組織。優化嘗試一檔一筆，依主題索引，並補登過去採用和未採用的做法。
- 2026-11-23 [FTF sampled RWA 探針的快取壽命](2026-11-23-ftf-rwa-probe-memo.md)：快取壽命。一次 FTF preparation 重用精確的探針證明；每次 evolve 仍選路、記帳、演化與發佈 capture。

- 2026-11-22 [舊實測數字當現況](2026-11-22-stale-measured-numbers.md)：量測新鮮度。引用數字附 commit、輸入與範圍，驅動決策前查被測路徑的後續改動；未完成列未走到的階段記 null。
- 2026-11-22 [模擬器接收雜訊只保留等向設定](2026-11-22-isotropic-receiver-noise.md)：刪除或收窄。模型 YAML 每個 receiver 改填單一 PSD，I 和 Q 獨立且變異數相同。
- 2026-11-22 [FTF 樣板的 projective floor](2026-11-22-ftf-projection-weight-floor.md)：精度取捨。FTF 樣板 floor 改為 `1e-5`；舊 history 模型下原設定第一點的 1,800 s 參考缺口，在 Markov 同模型驗收後不再適用。
- 2026-11-22 [ADC raw plan 由誰持有](2026-11-22-adc-raw-plan-acquire-owner.md)：快取歸屬。FTF 與 standalone single 的 acquire session 持有 ADC plan；換 batch 或結束就退役，舊 raw 可冷重算。
- 2026-11-22 [FTF projective 從真空續演化並改用 Markov chain](2026-11-22-ftf-projective-markov.md)：表示與資料格式。投影後逐 DC 態續演化，raw 保留單發配對，compact 只給逐事件邊際；獨立參考兩 floor 驗過 41／75 個 key。
- 2026-11-22 [模擬器的時間門檻與驗證成本](2026-11-22-simulator-speed-first-verification.md)：取捨原則。當時使用 180 秒目標，精度以 `1e-2` 為目標；優化先做局部對照，獨立參考與全表不再是每次落地關卡。
- 2026-11-22 [牆鐘工具的 worker 數與量測範圍](2026-11-22-wall-time-worker-policy.md)：量測條件。預設保留單 worker；非排他慢實驗子集按 CPU slot 明示 worker 數，以收據核對。

- 2026-11-21 [Projective 腔場改走 ADC flat span 合併解](2026-11-21-projective-span-reuse.md)：快取歸屬。Projective 與 MIST 都向 ADC 取 flat 二階矩與各 state 端點場，只在 exact key 命中時重用，FTF 仍逐 history 選態與建 shift；兩路共用 512 MiB，`CavityDrive` 只留 `cache=`；三個固定前綴同 seed density 相等、獨立端點解歸零。

- 2026-11-21 [FTF Stop：合作式檢查點與取消傳輸](2026-11-21-ftf-stop-checkpoints.md)：停止與取消。不設上限、設計上檢查間隔 5 秒；取消經 SimulationSoc request 生命週期到兩個 QICK round 入口，零輪拋既有停止例外；cold 首跑與 Engine 內剩餘區段未達 5 秒，另開 task 承接。
- 2026-11-21 [當輪必需 raw 的取消檢查與發佈](2026-11-21-required-raw-stop-publication.md)：取消與發佈（designer 代裁，使用者未覆核）。只在當輪需要 raw 時按呼叫借用 query；SoC 驗證 raw、通過最後查詢後才入帳並發布，statistics-only 與已完成輪 getter 保持 lazy。

- 2026-11-21 [FTF 合法讀出時序的 ADC image、drive admission 與到達時間缺口](2026-11-21-ftf-legal-readout-timing.md)：時序邊界。具名 ADC config 保留物理 image、無來源 raw 字按 DDS 同餘類解調，純 drive DAC 由 `drive_generators` admission；在途漏配明確拒絕，完整到達時間模型與 E2E 留待落地後排他重測。
- 2026-11-21 [FTF 讀出窗依裝置時間配對](2026-11-21-ftf-readout-device-time-pairing.md)：事件時序。ADC 擁有 `t−τ` 換算；planner 逐窗按裝置時間配對 block，最後相交 block 為 owner，投影與 ADC grid 不移動，跨 event 殘場仍是已知近似。
- 2026-11-21 [同 owner 的讀出窗共用脈衝歷史](2026-11-21-shared-owner-readout-membership.md)：事件成員（designer 代裁，待覆核）。同 ADC／owner event 用各窗交集 block 的 pulse 聯集，保留逐窗交集，不從其他 ADC 或 owner 借入 pulse。

- 2026-11-21 [raw ADC 頻率字改成編譯時拒絕](2026-11-21-raw-adc-word-reject.md)：刪除或收窄。無具名來源的 raw ADC 字編譯時拒絕，拿掉 DDS 同餘類與 Six 同餘選路。

- 2026-11-21 [MIST 腔場重用：ADC 擁有 flat span 的結果](2026-11-21-mist-cavity-reuse.md)：快取歸屬。ADC 的同一個 memo owner 收錄二階矩與各 state 端點場（合計 64 MiB、一次 composition），FTF 只選態與建 shift；mist/flux_dep 從 281.5 s 降到 163.3 s。

- 2026-11-21 [FTF projective 列身分與失敗定位](2026-11-21-ftf-projective-row-provenance.md)：識別作用域。FTF 給原點和 history 穩定的列 ID，在單次 preparation 內檢查碰撞，並用 capture ID 串起逐列來源。

- 2026-11-20 [FTF cavity Stark 反轉：從報錯改成警告](2026-11-20-ftf-stark-inversion-warning.md)：放寬或拒絕。依 ADR-0078 先例，Stark 位移讓相鄰態反轉時照原模型繼續算，每組反轉發一次警告並給 gain 共同縮放上限；反轉後模型仍算得出數字，所以否決保留報錯。

- 2026-11-19 [Evolution 輸出合法 density](2026-11-19-evolution-valid-density.md)：數值保證。純態經 DP5(4) 短段後最小特徵值為 −2.4e−6；engine 在 budget 內投影回合法 density，超出照數值失敗回報。

- 2026-11-18 [Full-drive gate 的 CPU 競品估價](2026-11-18-gate-direct-duration-price.md)：價格重校。原六組加上已量的 DRAG lab 窗重校 duration 端點，原 flat-top lab 連段仍未量測。

- 2026-11-18 [成形脈衝的 fallback 共用與 prepare key 計數](2026-11-18-shaped-fallback-key-reuse.md)：公開診斷。Engine 計入已丟棄的 guard 產品，提前解析保留 timeline 內的 Sambe spectral 共用與跨 timeline 的 period table 共用。

- 2026-11-18 [移除 private replay](2026-11-18-remove-private-replay.md)：刪除或收窄。FTF 與 single-fluxonium 不再逐位重算；debug capture 記下 Engine 輸入、選路與每輪的有效設定。

- 2026-11-18 [短 `dop853` row 改用 DP5(4)](2026-11-18-short-dop853-dp54.md)：公開診斷。`dop853` 保留原 family，公開 telemetry 分開計兩種 tableau 的實際端點；合成精度矩陣與真實 FTF 驗收分開。

- 2026-11-18 [`dop853` 的 OpenCL GPU backend](2026-11-18-ftf-gpu-dop853-backend.md)：新增執行後端。`dop853` 獨立 row 在有 FP64 OpenCL GPU 時成批交給 GPU，一條 timeline 一個 work-group，沒有 GPU 時逐位不變；replay 綁執行裝置，GUI 啟動在背景預編譯。

- 2026-11-17 [Capture 的唯讀體檢與 typed metadata](2026-11-17-capture-inspection.md)：唯讀報告。共用 owner schema 讀逐輪實錄，以 capture_file 對行，條件差異只標記、不重算。

- 2026-11-17 [Capture 的發布與 acquire 關聯](2026-11-17-capture-publication.md)：公開寫出。Writer 共用 cache root，以原子 manifest 發布和 capture_file 關聯三個 environment 的同次 telemetry。

- 2026-11-17 [Engine capture 的逐輪選路事件](2026-11-17-engine-capture-events.md)：公開錄製。每次 evolve 一個 ID，以既有凍結容器保存輸入，分開記 caller deferral、worker resolution 與 prefix 採用者。

- 2026-11-16 [Acquire wall 的分組與未量測資料](2026-11-16-acquire-wall-analysis.md)：彙總契約。只按 complete 分組，四個 wall 俱為數字才計入耗時，所有有效行保留 family 統計。
- 2026-11-16 [Single 與 six 的 acquire telemetry](2026-11-16-single-six-acquire-telemetry.md)：計時聚合。以 acquire 為寫入邊界，聚合各 source 與 projective span，直接沿用 binding 成敗。
- 2026-11-16 [FTF acquire 的演化與讀出 wall](2026-11-16-ftf-layer-walls.md)：分層計時。每次 Engine 與 ADC 工作各自累積 wall，跨 round 與 continuation 加總，失敗段保留已耗時間。

- 2026-11-16 [Acquire outcome 的作用域與編譯交接](2026-11-16-acquire-outcome-scope.md)：生命週期。Binding 以自身作用域判斷正常返回，mixin 只圈住 builder 計時，environment 接收同一結果物件。

- 2026-11-16 [共用 acquire telemetry writer](2026-11-16-shared-acquire-telemetry.md)：資料契約。共用八欄必填 summary，writer 明列投影，FTF projection 只放 namespaced extras。

- 2026-11-17 [DOP853 加速路線的證據與未驗項](2026-11-17-dop853-speed-route-summary.md)：路線比較。指定 Gaussian 批次的多 worker 有配對收益，取樣前導共享潛力未知，載波固定原型精度不合格，GPU 全 14 組比較延期未量。

- 2026-11-18 [FTF 模擬的子空間約化：各做法的結果與結案](2026-11-18-ftf-subspace-reduction.md)：刪除或收窄。Engine 內自動約化沒有便宜又嚴格的逐點 guard，維度由使用者以 retained_levels 宣告；單 tone 掃描 24 態合格，雙 qubit X90 長序列各做法都不划算。

- 2026-11-18 [FTF 動態維度：retained_levels 改為上限，每次 acquire 用 pilot 選維度](2026-11-18-ftf-dynamic-dimension.md)：預設行為改變。FTF runtime 以一條上限 pilot 讀 drop(k) 選維度，划算才驗算一次，Engine 不約化；F1 快 1.20 倍、雙 qubit 同時驅動快 2.49 倍。

- 2026-11-17 [DOP853 profile 場景的單列選取與 producer 接線](2026-11-17-dop853-profile-scenario-routing.md)：場景接線。Profile owner 先選名稱再建單一 Input，producer 附加宣告但不擴大 Pareto 預設選集，也不自動準備數值。

- 2026-11-17 [Projective read 的輸出改成 Markov chain](2026-11-17-projection-markov-chain-readout.md)：資料格式。Single／six 輸出每點初始分佈與逐 read 轉移表，integrated 用逐態 flow counts，floor 改成逐點 identity 近似；FTF 不動，決策見 [ADR-0078](../adr/0078-evolution-module-boundary.md#projective-read-的-markov-chain-表示)。

- 2026-11-17 [共用清單的 Pareto 選集與明確新增場景](2026-11-17-pareto-selection-and-scene-append.md)：選取契約。Pareto 自持預設名稱，明確更新才新增列，reference producer 只讀解所選單列。

- 2026-11-16 [常數雙 tone 的 secular guard 與唯讀診斷](2026-11-16-constant-multitone-guard-diagnostic.md)：適用域收窄。Guard 與公開 query 共用最終 admission fact，固定證據不外推到日常 FTF 或價格。
- 2026-11-15 [`common_frame` 的 coherent 門檻收窄與唯讀診斷](2026-11-15-common-frame-coherent-narrowing.md)：適用域收窄。只把既有 coherent 上限從 0.03 降到 0.0018，不加新指標；多拒絕的 5 個合成格與兩 tone 取樣列改由價格選 family。


- 2026-11-16 [凍結場景改為共用快取，Git 只留名稱與 hash 清單](2026-11-16-frozen-scene-storage.md)：資料入庫。數值不入庫，每筆單筆 gzip，Git 只留兩份小清單，重建不符就停。

- 2026-11-16 [Exclusion estimator 退休](2026-11-16-exclusion-estimator-retirement.md)：刪除或收窄。退休沒有非測試 caller 的 helper，收窄 guard 的耗時影響改量實際改動。

- 2026-11-16 [量測樣本時間的值域與成對比值檢查](2026-11-16-measurement-sample-durations.md)：值域取捨。Owner 原樣記錄有限非負秒數，成對區塊比值要求兩側總和為正。

- 2026-11-16 [成對 Engine 計時的 process、身份與統計](2026-11-16-paired-engine-timing.md)：跨 process 計時。Warm 長駐、first/cold 每筆新 child，四項 telemetry 校準共用一個模式。

- 2026-11-16 [共用模擬器計時與 reference 必填](2026-11-16-shared-simulator-measurement.md)：計時歸屬。Session、history 與 clock 共用 owner，consumer 保留 reference 與結果 schema，0 elapsed 原樣記錄。

- 2026-11-16 [Family 量表改為整批 cache-only](2026-11-16-family-pareto-cache-only.md)：介面退休。Consumer 整批載入後才量測，check 不求解；公開接縫提供穩定的每筆 record 身份，存放已遷移到共用快取與小清單。

- 2026-11-16 [模擬器場景凍結與參考解數值交接](2026-11-16-frozen-simulator-inputs.md)：資料交接。場景 producer 自持控制與模型，通用 codec 保留既有 key，獨立 solver 留在 tests 側。
- 2026-11-23 [Evolution 保留完整 collapse operators](2026-11-23-evolution-full-collapse-operators.md)：數值能力擴充。固定 H 的 CPU `dop853` 保留同一 jump 的相干轉移，不增加 FTF 專用介面。
- 2026-11-23 [凍結數值升到 physical v2](2026-11-23-frozen-physical-v2.md)：資料格式版本。保留無 jump 身份，另設場景與 capture 根。不支援完整 jump 的 solver 不得在 cache miss 時發布參考解。

- 2026-11-14 [single_tone_rwa 以 endpoint kick 保留，guard 改用 κ2](2026-11-14-single-tone-rwa-kick.md)：保留或刪除。隨機相位序列從最大 1.07e-2 降到 ≤2e-3；價格計入 kick 延後，L、I 的預設選路暫時比 floquet_period 慢超過 10%。

- 2026-11-16 [Floquet 的保留與價格重校](2026-11-16-floquet-kept-price.md)：保留或刪除。Committed 批次上 `floquet` 快 1.65–1.81 倍，保留並重校價格；三 family 排序等 `multitone_rwa` 修好。
- 2026-11-16 [Floquet 的拒絕成本與單次 evolve 內的 frame 重用](2026-11-16-floquet-refusal-cost.md)：拒絕成本。既有拒絕條件提前，一次 evolve 內 exact-key frame 只建一次；不把拒絕機率算進價格，不跨 key 或跨 evolve 保留 frame，不做建 frame 前的 trace 下界。

- 2026-11-15 [單段雙 tone benchmark 的量測邊界](2026-11-15-single-segment-benchmark-boundary.md)：量測邊界。專用 Engine factory 與既有 acquire 分開，模型與參考解仍由各自 owner 建立。

- 2026-11-21 [實驗 Run 牆鐘工具](2026-11-21-experiment-run-wall-time.md)：量測邊界。整個 Run 只用一支入庫工具記耗時，測試與 kernel 比較仍按層分開。
- 2026-11-21 [牆鐘報表的 guide 與輸入](2026-11-21-wall-time-coverage.md)：宣告歸屬。Adapter 交付 guide 路徑；工具固定模擬輸入，只計 Run 到終態。
- 2026-11-21 [牆鐘工具如何記錄 Run 失敗](2026-11-21-wall-time-run-failed.md)：失敗隔離。Controller 回報失敗時先寫列；idle 才續跑；exit status 區分未判定列與工具錯誤。
- 2026-11-21 [牆鐘報表如何重建中斷列](2026-11-21-wall-time-interrupted-rows.md)：刪除或收窄。派送前 journal 不能證明 Run 耗時，重建不填秒數或下限。
- 2026-11-22 [固定 FTF fixture 缺路由的十列](2026-11-22-wall-time-fixture-flux-routing.md)：量測邊界。固定 fixture 缺快磁通與共用 coil 的十列明示未量，不借示意係數或其他場景的 coil。
- 2026-11-23 [牆鐘工具量手選場景](2026-11-23-curated-wall-time-scenes.md)：資料與量測邊界。工具自備 FTF 設定和乾淨 point，由維護者選場景，派送現行實驗，不掃描 guide。
- 2026-11-21 [模擬器核心耗時：從完整 Run 讀，不建凍結場景](2026-11-21-simulator-core-time.md)：量測指標。從完整 Run 讀編譯、演化、讀出與 raw materialize；不完整列只報下限。記錄當時使用的 180 秒判定，以及不採錄製重播或手寫場景的理由。
- 2026-11-23 [模擬器速度改按每格點](2026-11-23-simulator-speed-per-point.md)：量測指標。掃描與逐次評估用平均每點 0.1 秒，無軸 shots 保留 180 秒；縮減場景的外推與實測分欄。

- 2026-11-15 [模擬器測試與量測的分層](2026-11-15-simulator-test-measurement-layering.md)：分層。Kernel、dispatch、序列、前端與 environment 分開驗，退休 acquire e2e 量測，決策見 [ADR-0078](../adr/0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)。

- 2026-11-15 [t6 只保留 direct](2026-11-15-t6-channel-removal.md)：刪除或收窄。移除代表性工作負載未選用的 channel 做法與價格分支，保留 direct 與既有 family。

- 2026-11-14 [Sampled multitone 的過渡期拒收](2026-11-14-sampled-multitone-interim.md)：收窄 admission。兩個 multitone family 暫停 sampled 列，保留 constant pairs 與既有 fallback 定價。
- 2026-11-18 [Sampled multitone RWA 的實測放行](2026-11-18-sampled-multitone-rwa-admission.md)：實測放行。同形 sampled 雙 tone 列按群抽查並計價，24 點正例快 3.149 倍，拒收照原選路。
- 2026-11-19 [成形雙 tone 的共同選路](2026-11-19-shared-multitone-pricing.md)：候選排序。CPU 重疊群按總 CPU 秒選路，跨群的 gate 建構按使用次數分攤並在建構前重核。

- 2026-11-14 [RB 型相位序列作為固定量表輸入](2026-11-14-random-gate-sequence-workloads.md)：場景擴充。離線固定 phase prefix 與 recovery，constant／Gaussian 共用，不加入 runtime RB 代數。

- 2026-11-14 [common_frame 的步長成本歸 dispatch](2026-11-14-common-frame-step-pricing.md)：成本歸屬。移除 kernel 的 2 ns 下限，dispatch 依 first-order 步數比較成本，數值 guard 決定適用域。

- 2026-11-13 [Family 支配的定義：容許範圍內只比耗時](2026-11-13-family-dominance-within-tolerance.md)：取捨原則。合格 family 只比耗時，刪除前仍須同型態長序列證據與使用者點頭。

- 2026-11-14 [Family 量表的單場景重量與摘要](2026-11-14-family-pareto-remeasurement.md)：資料入庫。參考解與完整結果跨 worktree 共用本機儲存，摘要入庫，比對不判跨時段快慢。

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
- 2026-11-14 [t6 的 direct 加速策略與職責歸屬](2026-11-14-t6-direct-strategy.md)：加速策略與職責歸屬。`t6` 依完整 acquire 的 key 列數在 channel 與 H0 旋轉座標 direct 積分間選擇，direct 容差跟 budget，定價與選擇歸 dispatch helper。

## Program 與 timeline 編譯

- 2026-10-31 [Timeline 編譯：變數模型、通用 join 與 s14 事件時間](2026-10-31-timeline-variable-model.md)：Timeline 合流採共用 join，範圍證明改用 Abs 表示，現行決策見 [ADR-0083](../adr/0083-program-macros-and-timelines.md)。
- 2026-11-20 [Two-qubit RB 在板上掃 depth：提早結束本發與控制結構的完備性](2026-11-20-early-exit-and-control-flow-completeness.md)：控制結構完備性。現有 repeat／branch 無法結束本發；spike 用 CondJump 加手抄狀態可行但脆弱，改為抽出 Branch 與 Repeat 共用的編譯期狀態，再補「提早結束本發」，方向見 [草稿](../adr/draft/control-flow-completeness.md)。
- 2026-11-20 [Timeline 共用的編譯期狀態：PathState 與 capture／restore](2026-11-20-timeline-path-state.md)：共用抽象。Branch 與 Repeat 改用 frozen `PathState`，capture／restore 一次處理全部隨路徑變化的狀態，合併規則留在各結構；不按 timeline 切，也不用指標切換，因為狀態屬於路徑且 QICK macro 就地讀寫 program 狀態。
- 2026-11-20 [Timeline 編譯器推算每發 ADC 讀取數](2026-11-20-timeline-read-counts.md)：路徑會合規則。編譯器在 macro 包裝層計數 ADC trigger，每個 Branch 出口逐 ADC 要求各 case 相同，固定 Repeat 乘次數，變動次數 Repeat 不能讀；自動寫入每發讀取數，`set_reads_per_shot` 只核對；否決逐條路徑比對整發總數。
- 2026-11-20 [CZ 優化候選用記憶體 arb 播放](2026-11-20-in-memory-arb-waveforms.md)：執行期波形。候選直接交 I/Q 樣本給 program，不寫檔；永久保存仍用既有 arb 資產。
- 2026-11-20 [Timeline 往後跳轉與 two-qubit RB 分塊](2026-11-20-timeline-forward-jumps.md)：控制流範圍。跳轉限於 body 最外層，label 合併路徑狀態；RB 分塊的代裁後來由使用者取消。
- 2026-11-20 [板上 RB 程式改寫](2026-11-20-rb-depth-program.md)：刪除或收窄。每個 seed、arm 保留一支 program，recovery 內嵌，PMEM 超量就報錯。
- 2026-11-20 [RB 取數歸約與衰減擬合品質](2026-11-20-rb-round-reduction-and-fit-quality.md)：資料格式。取數端歸約 rounds，Result 以 (seed, arm, depth) 開頭，two-qubit fit quality 回報衰減長度相對誤差。

## Measure UI

- 2026-11-23 [Heralded RB 與 conditional phase 合併顯示 gg、ee](2026-11-23-heralded-pooled-display.md)：圖形呈現。兩個 heralded 入口合併主圖與 Live，分組診斷及原分析不變。
- 2026-11-23 [CZ phase 的 contrast 標準誤](2026-11-23-phase-contrast-sem.md)：圖形呈現。Live 與 Analyze 傳遞獨立 X/Y 投影的 shot SEM，恰零時取最大方向。
- 2026-11-23 [Conditional phase 與 Local Z 的圖照推導順序](2026-11-23-phase-figures-physics-order.md)：圖形呈現。主圖從 Bloch 向量、累積相位到條件相位；診斷量另放；Live 只畫已完整頻率。
- 2026-11-23 [Singleshot T1 以振幅分 T1／T1b](2026-11-23-singleshot-t1-amplitude-naming.md)：讀圖語意。e→e 曲線振幅大的模是 T1、小的是 T1b；兩版都提議寫回主模。
- 2026-11-23 [Single-shot T1 的兩個衰減時間與 tone Stop 初值](2026-11-23-singleshot-t1-relaxation-times.md)：讀圖語意。Superseded，由[振幅命名](2026-11-23-singleshot-t1-amplitude-naming.md)取代；Stop 初值仍取五倍。
- 2026-11-23 [一維反演量的誤差棒與圖上事實](2026-11-23-inverted-error-bars-and-figure-facts.md)：圖形呈現。可傳遞 shot 共變異的一維反演量也畫棒；圖呈現品質值，寫回建議留在 Writeback。
- 2026-11-23 [已存統計量的第一層誤差棒](2026-11-23-errorbars-tier1-saved-stats.md)：範圍分層。只畫已存矩可推的棒（reset、chain joint residual、readout power），不改取數與存檔；合併矩不扣 round 間漂移。
- 2026-11-23 [MIST 距離的一階誤差棒](2026-11-23-mist-trace-propagated-sem.md)：圖形呈現。同批六矩傳遞平均 IQ 距離的 shot SEM，Live 與 Analyze 同式。
- 2026-11-23 [IQ 六矩共用原始矩捨入界](2026-11-23-shared-iq-rounding.md)：數值判準。矩陣與線性投影共用原始矩門檻；不可估的點留中心、只略依賴它的棒。
- 2026-11-23 [讀出矩陣奇異就當作沒有矩陣](2026-11-23-singular-matrix-as-absent.md)：Analyze 失敗語意。奇異或溢位的保存矩陣等同缺席，未校正照常 fit、摘要與 Writeback，原因記 log。
- 2026-11-23 [無校正的 Raman／STIRAP 分類圖與摘要](2026-11-23-classified-raman-stirap-analysis.md)：Analyze 失敗語意。缺校正時畫保存的分類比例、診斷留 null，STIRAP 另報分類 f 峰。
- 2026-11-23 [一維讀出校正無解時保留分類比例](2026-11-23-singular-readout-analysis.md)：Analyze 失敗語意。Superseded，由[無矩陣等價](2026-11-23-singular-matrix-as-absent.md)取代。
- 2026-11-23 [圖磚沿用存檔幾何](2026-11-23-figure-fixed-geometry.md)：圖形呈現。各面以 12×9 in 排版後等比嵌入。R7 只在固定幾何的 PNG 驗版面，Data 保留縮圖選圖與 `View figure`。
- 2026-11-22 [在 Measure Data 頁讀原圖](2026-11-22-data-figure-reading.md)：圖形呈現。記錄當時用縮圖選圖、在同頁捲動閱讀的做法。現行圖磚與閱讀面見 [Figure presentation](../../lib/zcu_tools/gui/app/measure/README.md#figure-presentation)。
- 2026-11-22 [Live-only 實驗的 figures-only Analyze](2026-11-22-figures-only-analyze.md)：Analyze 能力。四個 leaf 只從明確來源重畫具名圖，不增加數值結果或寫回。
- 2026-11-22 [非均勻熱圖在建構時選色盤](2026-11-22-nonuniform-heatmap-cmap.md)：繪圖介面。作者在建構期傳入具名 cmap，預設與其他熱圖行為不變。
- 2026-11-22 [GE 寫回保護先於診斷圖交付](2026-11-22-ge-gate-split-from-diagnostic.md)：交付切分。窄磚上診斷圖標籤仍相碰，寫回初選的中心檢查先獨立落地，直方圖與擬合 SNR 整批延到續作。
- 2026-11-22 [GE 診斷圖保留四格](2026-11-22-ge-diagnostic-layout.md)：圖面取捨。assignment 移到兩列之間；當時採多數準備取色，後記說明改為銳化混色。
- 2026-11-22 [GE Post 的準備顏色與分類邊界](2026-11-22-ge-post-preparation-majority.md)：讀圖語意。當時採準備共格取色；後記說明回到分格，分類邊界仍與上色分工。
- 2026-11-23 [GE Post 以分類結果上色](2026-11-23-ge-post-classified-color.md)：讀圖語意。Post 保留準備分格，顏色改表示各發的分類；Analyze 保留準備色。
- 2026-11-23 [Mux GE Fit 用準備色，圖例標出 probe 來源](2026-11-22-mux-fit-preparation-color.md)：讀圖語意。Fit 與 Diagnostic 同用準備色；從 Fit 圖例追溯 Live 的取數列。
- 2026-11-22 [實驗 guide 只寫操作者需要的內容](2026-11-22-guide-content.md)：文件範圍。改寫規則並清理 111 份 guide，通用操作移到共用頁，排錯與可重用判讀留在知識庫。

## Cfg 與表單

- 2026-10-13 [Measure 表單重設計：從 builder 重抄到 core cfg 推導](2026-10-13-measure-form-redesign.md)：表單形狀從 core cfg 推導，說明與初值各有 owner，決策見 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-13 [表單選項的分層：生成器給選項，widget 照清單畫](2026-10-13-form-options-layering.md)：生成器給選項而 widget 只呈現，現行分工見 [ADR-0065](../adr/0065-cfg-editing.md) 與 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-17 [Measure Form 的校準初值與 sweep 模式切換](2026-10-17-measure-form-initial-values-and-sweep-mode.md)：校準初值在開 Form 時算成數字，缺值、無效值與故障分流，決策見 [ADR-0073](../adr/0073-measure-form-derivation.md)。
- 2026-10-29 [Singleshot 分類參數放進 readout cfg](2026-10-29-readout-classifier.md)：Classifier 隨 readout cfg 保存，作者明確傳入 acquisition，決策見 [ADR-0074](../adr/0074-flat-readout-module.md)。
- 2026-11-20 [v2 實驗命名與顯示：先對齊 two_qubit 的圖](2026-11-20-experiment-naming-and-display.md)：命名取捨。全 v2 共用分類，先定使用者看得到的名字；第一批只對齊 two_qubit 的圖與 guide 首標，不改 key、tag 或 cfg。
- 2026-11-21 [表單預設超過板子限制，錯誤解析放知識庫](2026-11-21-form-board-limits.md)：文件邊界。指南描述實驗行為，知識庫說明板子限制與錯誤；修正 `reset_check` 的 gain 和 CPMG 的 Minimum time 預設值，決策見 [ADR-0073](../adr/0073-measure-form-derivation.md)、[ADR-0075](../adr/0075-agent-interface-layers.md#判讀依據放知識庫)。
- 2026-11-21 [Gain 固定範圍：表單只驗純量，掃描留給 Run](2026-11-21-form-gain-range.md)：刪除或收窄。表單以既有 annotation 限制純量 gain；一般掃描保留 `SweepCfg`，超界點在 acquisition 前被拒。
- 2026-11-21 [readout 時序預設：固定長度差與光子衰減等待](2026-11-21-readout-timing-defaults.md)：初值與預設值。範本前後各留 0.015 µs，chain／Rπ length 改用可編輯長度差（schema 6.0／4.0），mux post_delay 種子取 5/(2π·κ)；範本差值讀注入的 catalog。
- 2026-11-22 [實驗呈現規範：圖、摘要與讀者檢查](2026-11-22-presentation-rules.md)：呈現規範。圖只放理論重點，採集設定進摘要，對數值大小敏感的一維資料畫 error bar，規範集中在 `figure-style.md`。
- 2026-11-22 [量測點何時連線與 spectrum 的 error bar](2026-11-22-data-point-lines.md)：呈現規則。無擬合曲線的一維圖連線、有擬合只畫點；有 shot 資料就畫 SEM，spectrum 不再例外；wrapped phase 只斷跨切口、恰半圈與缺值的線段。
- 2026-11-22 [Analyze 摘要與逐點誤差棒的責任分工](2026-11-22-shared-presentation-facts.md)：呈現責任。GUI 與 MCP 共用 adapter 摘要行，analysis 計算 shot SEM，實驗 core 寫標題與圖例。
- 2026-11-23 [平均 IQ 圖的投影慣例](2026-11-23-mean-iq-plot-projection.md)：呈現規則。同類實驗沿用同一 IQ 轉換，一維預設用主軸投影，二維 flux／頻率圖逐 bias 扣背景再取幅值。
- 2026-11-23 [Live 圖不顯示進度計數](2026-11-23-live-no-progress.md)：圖形呈現。進度只由 GUI progress bar 顯示；Live 保留描述資料與判讀的標註。
- 2026-11-22 [Two-qubit mux 綁定與編輯後的檢查](2026-11-22-two-qubit-mux-binding-checks.md)：驗證位置。Open／Reset 驗有序元件連結，提交時按同次 capture 驗實際 readout tone 數。
- 2026-11-23 [掃描初值至少 31 點、用奇數](2026-11-23-scan-defaults-min-31-odd.md)：預設值。等距掃描初值每軸 ≥31 點，以中心為準者用奇數；只改初值不擋輸入，離散 protocol 序列除外。
- 2026-11-23 [表單預設引用 library](2026-11-23-form-ref-not-copy.md)：表單預設。只在表單改了 Run 不會覆寫的欄位時複製，試調先 Convert to inline。

## 模組邊界

- 2026-11-15 [模擬器封閉成子模塊](2026-11-15-simulator-closed-submodule.md)：依賴邊界。`simulate` 不 import `zcu_tools` 其他 package，邊界型別歸它，裝置模型由使用方經介面聲明，決策見 [ADR-0072](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明)。
- 2026-11-17 [封閉 simulate 的搬遷順序](2026-11-17-simulate-closure-expand-order.md)：搬遷順序。被 contract 擋住的暫時轉出改放在允許的方向（新位置轉出舊定義），分批改 importer，最後一票才搬定義，main 不出現 re-export。
- 2026-11-14 [Lab 依賴只經 assembly 入口](2026-11-14-lab-dependency-boundary.md)：依賴邊界。組合根與開發工具只直接 import assembly，migrations 保留格式遷移所需的直接依賴，決策見 [ADR-0072](../adr/0072-module-boundaries.md)。
- 2026-10-20 [框架去具體化：alias、模板、predictor 與一晶片一 qubit 假設](2026-10-20-framework-decoupling.md)：領域 alias、模板與模型認知留在 zcu_lab，框架的 owner 邊界見 [ADR-0072](../adr/0072-module-boundaries.md)。
- 2026-11-22 [Two-qubit 的身分、定位與中性位置名](2026-11-22-two-qubit-identity-and-neutral-positions.md)：職責歸屬。Adapter 解析身分並傳定位數值，分析 options 接收標籤，位置沿用 `qubit1/qubit2`。
- 2026-11-22 [頻譜擬合模型的求值 owner](2026-11-22-spectrum-fit-evaluation-owner.md)：擬合邊界。Fitting 重用同次係數，實驗 caller 決定窗內網格與呈現。
- 2026-11-23 [Program 的模擬邏輯只在 mixin 內](2026-11-23-program-simulation-mixin.md)：依賴邊界。單輪模擬估時移入模擬子包，lab 不再傳模擬專用線路要求。
- 2026-11-23 [哪些取捨要寫進 ADR](2026-11-23-adr-inclusion-criteria.md)：文件組織。可能約束多個底層模組、決定子系統共同架構、難以撤回或需記錄否決理由，符合任一條就寫進 ADR。

## 測試與驗證

- 2026-11-17 [GUI teardown 的可見視窗 guard 與 owner 清理](2026-11-17-gui-teardown-visible-window-guard.md)：測試護欄。GUI 測試 teardown 後仍可見的 top-level 視窗讓該測試失敗；guard 立即取樣，不先交付 DeferredDelete；fixture 只退休自己擁有的視窗；產品退休的 form row 明確隱藏。
- 2026-11-16 [Tools 標準化：補工具缺口，不搬目錄、不加硬閘](2026-11-16-tools-standardization.md)：工具標準化。品質工具補 `-h`／argv 與 docstring，加只報告的 docstring 現況列與 README 正向規則；不搬目錄、不加硬閘。
- 2026-11-16 [實驗 probe 的 snapshot 保存與比較](2026-11-16-probe-snapshots.md)：工具範圍。保存與比較留在 tools，階段不寫 snapshot，環境清理由獨立 context manager 收尾。
- 2026-11-15 [Doc relations 略過 fenced code 示例連結](2026-11-15-doc-relations-fenced-links.md)：檢查器誤報，修正 checker 對示例的判讀，保留正文連結與 ADR 檢查。
- 2026-10-16 [Agent 自行做 UI e2e：元件驅動情境](2026-10-16-agent-ui-e2e.md)：Agent 經使用者可操作的元件驗證可見行為，使用者走查判品味。
- 2026-11-22 [Families 部分交付與代表 R0 缺口](2026-11-22-families-partial-delivery.md)：驗收範圍。已驗 leaf 與 mux reducer 進版；Coherence 四葉、MIST、JPA、Fast-flux、Chain 等代表 R0 未驗時如實保留缺口，不寫成通過。

## 存檔與資料格式

- 2026-10-14 [量測成果 Artifact、設定快照與讀回的推導](2026-10-14-artifact-persistence.md)：設定快照正本只在 data.h5，保存與讀回決策見 [ADR-0063](../adr/0063-persistence-ownership.md)、[ADR-0069](../adr/0069-experiment-core-and-frontends.md) 與 [ADR-0067](../adr/0067-gui-application.md)。
- 2026-10-23 [Store 讀取效率與寫入慣例](2026-10-23-store-read-efficiency.md)：保留分組提交，以增量讀和內容 hash 快照減少重複驗證。
- 2026-11-16 [range_find 的 writeback：沒有就建立，有就覆寫](2026-11-16-range-find-writeback-upsert.md)：寫回目標缺元件。參數列帶選填 create_kind，Apply 在同一次 edit 先建元件再設值；不加列種類或入口，決策見 [ADR-0008](../adr/0008-cfg-editor-session.md)。
- 2026-11-17 [Global／point 參數型別收斂與文件組](2026-11-17-global-point-parameter-types-and-document-set.md)：型別收斂。Global 與 point 共用 `ParameterView` 等一套型別，以 `point=None` 選 global；三檔合稱文件組由 session 持有，每種檔一個形狀相同的開檔入口，決策見 [ADR-0063](../adr/0063-persistence-ownership.md#同一種資料不因-global-或-point-分兩套)。
- 2026-11-17 [Aliases 綁檔物件的讀取、refresh 與候選准入](2026-11-17-alias-file-cached-observation.md)：公開契約語意。物件持有一次觀察，只有 refresh 讀檔；session 以 opener 開候選，語法通過才與表一起換上。
- 2026-11-18 [Results Browser 開窗與背景索引](2026-11-18-results-browser-open-speed.md)：背景工作邊界。開窗逐批交付，隱藏不重掃；唯讀索引不算 operation，見 [Measure README](../../lib/zcu_tools/gui/app/measure/README.md#owners)。
- 2026-11-19 [Point 與 library 資料流的摩擦](2026-11-19-point-library-frictions.md)：便利功能取捨。Ramsey 頻率寫回、回滾、過時 tab 都不改，crosstalk 只做純函式；小功能要多次遇到才加，原則見 [ADR-0075](../adr/0075-agent-interface-layers.md#先做簡單設計再按需加法)。
- 2026-11-19 [單 qubit RB 與 IRB 合併成 twotone/rb](2026-11-19-single-qubit-rb-irb-merge.md)：實驗合併與資料格式。`target_gate=None` 只跑 reference，指定 gate 才跑兩臂；Result 改為 (round, seed, arm, depth)；`twotone/irb` 移除不留 alias，舊 run 只保留原檔。
- 2026-11-20 [用 writeback 發布 CZ 的 arb 資產](2026-11-20-arb-asset-writeback.md)：資產發布。Apply 先發布不覆寫的 entry 資產，再寫引用該 key 的 library 提案。
- 2026-11-22 [Two-qubit 新 native 軸名與舊格式](2026-11-22-two-qubit-native-axis-cutover.md)：資料格式。只有模擬測試資料的家族不新增不相容舊 layout 的 reader，原檔保留。
- 2026-11-22 [Amp Rabi 的 shot moments](2026-11-22-amp-rabi-shot-moments.md)：資料格式。同批校正 IQ 保留既有平均，native 可選 moments；驗所有現有變數後，Labber 仍只匯出原單 log。
- 2026-11-22 [Mux 逐 tone 校正 IQ 矩與 R11](2026-11-22-mux-iq-moments-and-r11.md)：資料格式。Reducer 按參與 tone 順序計算完成輪的同批校正六矩，保留既有平均與舊檔。
- 2026-11-23 [Labber scalar 匯出的索引軸](2026-11-23-labber-scalar-index-axis.md)：資料格式（已撤回）。原做法是替無軸 scalar 加一點索引軸。
- 2026-11-23 [CZ 最佳化的 Labber 選集](2026-11-23-cz-optimize-labber-selection.md)：資料格式（superseded）。匯出 `params` 矩陣的做法由 [具名 Native 與 Labber 欄位](2026-11-24-cz-optimize-native-labber-format.md) 取代。
- 2026-11-23 [Coherence 的 shot SEM 只隨當次 Run 交接](2026-11-23-coherence-run-only-shot-sem.md)：資料入庫（superseded）。只在記憶體交接矩的保存決策已由 [Native 保存六矩](2026-11-23-coherence-native-shot-moments.md) 取代。
- 2026-11-23 [Coherence 的 shot 矩寫入 Native 檔](2026-11-23-coherence-native-shot-moments.md)：資料入庫。三個 leaf 可選保存同次六矩，重載後重算 SEM；Labber 匯出與三個歷史輸入契約不變。
- 2026-11-23 [舊 MIST gain Labber 的同版轉換](2026-11-23-mist-legacy-labber-equivalence.md)：歷史格式。固定原單 log 宣告，兩種 profile 證明等價才保留版號。
- 2026-11-24 [掃頻 drive cfg 的值域擴張升 minor](2026-11-24-swept-drive-cfg-minor.md)：資料格式。五個入口保留完整來源值，舊等頻 cfg 可由新版載入；舊版讀新異頻檔保留結果但不還原 cfg。
- 2026-11-24 [CZ 最佳化的具名 Native 與 Labber 欄位](2026-11-24-cz-optimize-native-labber-format.md)：資料格式。逐欄保存參數及單位；cfg 無法還原時保留量測和分析，舊格式用原版本開啟。

## 量測策略

- 2026-11-22 [K-cycle chain 只留主擬合](2026-11-22-chain-joint-fit-only.md)：刪除或收窄。移除巢狀擬合與 `CycleCheck`，用白化殘差和 live 方向投影判讀，不讓第一讀估計或投影缺席蓋掉 joint 結果。
- 2026-11-22 [Fast-flux 頻譜擬合失敗](2026-11-22-fastflux-distortion-fit-failure.md)：失敗語意。保留整次 Analyze 失敗，不用部分成功圖掩蓋不可辨識的掃描。
- 2026-10-27 [T1/T2 over flux 量測策略：哪些機制有效](2026-10-27-coherence-strategy-levers.md)：自適應 flux 選點無增益，穩健 delay 取樣讓每條曲線的 σ 約為原來 0.73 倍。
- 2026-10-29 [Heralding：取數層 post-selection，不做 FPGA active reset](2026-10-29-heralding-post-selection.md)：在 host 取數層依 herald 篩選再平均，不做 FPGA active reset，決策見 [ADR-0074](../adr/0074-flat-readout-module.md)。
- 2026-10-30 [Readout 優化指標：用兩次讀出的一、二次矩算 SNR 與 QND](2026-10-30-readout-optimization-metrics.md)：兩次讀出的一、二次矩估 SNR 與 QND，先畫 Pareto 前緣由人選。
- 2026-11-02 [Heralding：herald／signal 二階矩自校準的轉移矩陣與指標 C](2026-11-02-herald-transfer-matrix.md)：IQ 改二階特徵，從 herald 與 signal 的 36 個總和自校準，只用 C = 1 − tr J 分析，不需要 classifier。
- 2026-11-03 [重複（A + readout）讀出鏈：效率、讀出 MIST 與三視角分解](2026-11-03-repeated-readout-chain.md)：重複讀出讓精度約隨 √N 提高，讀出 MIST 沿鏈累積限制 N；三視角分解遇讀出躍遷時沒有優勢。
- 2026-11-03 [讀出鏈的 trace 不變量：低 SNR 下比較 A 變化的通用指標](2026-11-03-readout-chain-trace-invariant.md)：連續讀出鏈存 lag 0…L 共變異，每點以 tr(C₀⁻¹C_ℓ) 的非迭代等比擬合輸出 Â 與 λ̂，不需校準、態數、參考或門檻；多態分析只留給 MIST 類特殊需求。指標形式已由 [讀出鏈的資訊上限](2026-11-04-readout-chain-information-limits.md) 取代。
- 2026-11-04 [讀出鏈的資訊上限：λ 指標、記錄層與分析層的損失、自校準](2026-11-04-readout-chain-information-limits.md)：顯示只畫 λ；讀出已知時讀出鏈在等時間下全面勝過 relax 實驗，存 lag 0…4 只損失 ≤ 18%，損失集中在分析層；改走全域兩態、兩通道的自校準估計。
- 2026-11-15 [Two-qubit RB 的兩組都同時打單 qubit gates](2026-11-15-two-qubit-rb-simultaneous-gates.md)：並行排程。普通與heralded兩版的reference／interleaved都用每qubit時間線，在CZ共同同步，保留原frame與reset政策。
- 2026-11-20 [Herald 改為 reset 模塊與逐 shot 遮罩](2026-11-20-herald-reset-shot-masks.md)：推翻放置選擇。Herald 改成既有 reset 欄位的模塊選項，取數層以逐 shot bool 遮罩篩選、所有下游統一收遮罩，tracker 改成顯式 reducer，FTF 沿用逐 shot 抽樣，決策見 [ADR-0074](../adr/0074-flat-readout-module.md#herald-reset-與逐-shot-遮罩)。
- 2026-11-20 [只為共用參數保留獨立 heralded 實驗](2026-11-20-heralded-experiment-pruning.md)：刪除或收窄。只保留 RB 與 conditional phase 的獨立入口，其餘七個用 mux herald reset 篩選 gg。
- 2026-11-23 [以準備後的 herald 分支量條件頻譜](2026-11-23-heralded-conditional-spectroscopy.md)：實驗入口新增。先做名目 I／Xπ 準備，再依實測 herald 類別畫四支條件頻譜。
- 2026-11-19 [Integer fluxonium 的 MIST：與 half-flux 的差異及 erasure check 的注意事項](2026-11-19-integer-fluxonium-mist.md)：物理研究。Φ=0 的 e/f 是 parity 相反的計算態加 erasure 態，MIST 線成對、e/f 不經 k=0 交換但 10⁻³ Φ₀ 偏磁即破；check 宜讓 g、f 都暗，最糟後果是 e→g/f 或漏到不被標記的高態；重複 check 的 Stark 相位與磁通雜訊會相乘累積；An 的 check 期間 T1(e→g) 變短無法以閉系統 MIST 或平滑浴解釋。
- 2026-11-20 [T1 二階矩實驗以 t1_cov 恢復](2026-11-20-t1-cov-restore.md)：恢復已刪除的概念。2026-11-09 誤刪的是二階矩方法，不是 post-selection；先恢復 T1，改稱 cov，讀出叫 pre_readout／post_readout，加可選脈衝 reset，舊資料不載入。
- 2026-11-20 [Two-qubit RB 在 Clifford 邊界對齊兩條時間線](2026-11-20-rb-clifford-barrier.md)：排程修正。2026-11-15 的並行排程只在 CZ 同步，reference 組相鄰 Clifford 會重疊，interleaved 組不會；改為兩組都在 Clifford 邊界 join，interleaved 時序不變、reference 變長，新舊資料不混比。
- 2026-11-20 [RB 每個 depth 的誤差棒](2026-11-20-rb-errorbars.md)：圖上診斷。畫 seed 間標準差，不加 shot noise 參考線，不改 Analyze 回傳；RB 與 IRB 數值工具合併。
- 2026-11-21 [Herald 後的等待由 readout 的 post_delay 擁有](2026-11-21-herald-wait-readout-post-delay.md)：欄位歸屬。Herald 後的等待由 readout 的 `post_delay` 擁有，不加獨立欄位；guide 寫需求、症狀與驗證方法。
- 2026-11-21 [Mux readout 的 post_delay 從五個光子衰減時間起步](2026-11-21-mux-readout-post-delay-seed.md)：表單初值。Mux Open／Reset 用 `5 / (2π · κ)` 給起點，操作者再用 herald 後的名目 Xπ 驗證。
- 2026-11-21 [讀出鏈的 pulse 長度與間隔](2026-11-21-chain-readout-spacing.md)：讀出時序。九個 chain 實驗等前一個 pulse 結束加 post-delay，不另等 ADC 關窗。該篇的 pulse 長度公式已局部取代。
- 2026-11-21 [Chain length 掃描固定 pulse 與 ADC window 的長度差](2026-11-21-chain-length-fixed-margin.md)：讀出時序。每點 pulse 長度為當點 ADC window 加上可編輯的長度差，不在 cfg 另填飛行時間。
- 2026-11-21 [Rπ length 沿用讀出模組的長度差](2026-11-21-rpi-length-fixed-margin.md)：讀出時序。Rπ 的 ADC window 掃描沿用可編輯的模組長度差，保留交替脈衝序列與結果形狀。
- 2026-11-21 [讀出範本在 ADC 窗口前後各留 0.015 µs](2026-11-21-readout-template-window-margin.md)：預設時序。範本、DPM 寫回與普通 onetone 表單共用 0.03 µs 的長度差。
- 2026-11-23 [MIST single trace 保留平均 IQ 的位移大小](2026-11-23-mist-single-trace-observable.md)：讀圖語意。Live 與預設 Analyze 用同一掃描前段複數參考，畫幅值而不套用 shot SEM（不畫棒已由[新紀錄](2026-11-23-mist-trace-propagated-sem.md)取代）。
- 2026-11-24 [CZ 最佳化只搜尋等長的 arb 脈衝](2026-11-24-cz-optimize-search-space.md)：搜尋空間。起點照原樣播放；後續只搜尋等長 arb，複數峰歸一，共同倍率與兩個 local Z 一起調。
