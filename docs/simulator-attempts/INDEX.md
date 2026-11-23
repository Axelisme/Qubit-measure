# 模擬器優化嘗試

**讀者：** framework-dev、agent。依主題查找模擬器的效能嘗試、採用狀態與實測範圍。

## 收錄規則

- 每次以速度或資源用量為目標的模擬器嘗試，在落地或放棄時新增一檔，並在下方選一個主題加一列。部分採用、暫停或仍在調查的嘗試也可先登記，結案時更新同一檔。單純的介面或資料格式變更不收。
- 已有討論就連到原文，不搬寫比較過程或量測表。依據只在未入庫的工作檔時，把結論與判斷所需的數字寫進正文，並說明原始工作檔未入庫。
- 索引每列放日期、嘗試檔連結、狀態、一句做法和一個代表數字。每個實測數字旁標明量測 commit、輸入與範圍，例如單輪或總和、實測或外推。缺少可追溯的量測時，可列有來源的設計值、門檻或算例，並標明用途。資料不全就寫明缺什麼，不以實作 commit 代替量測 commit。
- 狀態記錄這次嘗試的結果，不描述目前的模組契約。若後續嘗試取代這次做法，在 `superseded_by` 填後續嘗試的相對路徑。

## 單檔模板

檔名用 `YYYY-MM-DD-slug.md`。狀態選 `adopted`、`rejected`、`partial`、`paused` 或 `investigating`。`commit` 填實作 commit；原型未入庫則填量測所用的 commit，並在正文說明原型未入庫。`discussion` 填討論的相對路徑，沒有就填 `none`；正文可直連原討論的段落。

```markdown
---
date: YYYY-MM-DD
status: investigating
commit: <commit>
modules:
  - <module path>
discussion: none
superseded_by: null
---

# <嘗試名稱>

**讀者：** framework-dev、agent。<這份紀錄的用途>。

## 核心思路

<想省掉哪一項工作，以及做法。>

## 實測

<連到原討論，或記錄數字、量測 commit、輸入、範圍與限制。>

## 為何採用或否決

<說明結果與決定，或連到原討論的理由。>

## 留下的線索

<後續可重用的做法、待驗證的範圍或不必重試的條件。>
```

## Family 與 guard

- 2026-11-23，[Sampled multitone RWA 探針提早判死](./2026-11-23-sampled-rwa-probe-early-rejection.md)，`rejected`。中途超標就退出會誤拒端點合格的列。D=4 正例在 10% span 的密度差為 `2.117e-4`，高於 `1e-4` 門檻（`8cd18afcd`，1.5 ns 的單一 spot-edge probe）。
- 2026-11-22，[讓被拒的 sampled span 改走 common-frame](./2026-11-22-sampled-span-common-frame.md)，`rejected`。強制放行和加入包絡斜率修正都比留在 DP5(4) 慢。強制放行 0.877 s，DP5(4) 0.782 s（`47fbcf5e8`，heralded RB 的 400 ns 雙 tone 末段，單列 apply 牆鐘）。
- 2026-11-19，[成形雙 tone 共同估價](./2026-11-19-shared-multitone-pricing.md)，`adopted`。以群總 CPU 秒數比較 gate 與 sampled RWA，建 gate 前按剩下的用量重算回本條件。原討論假設建構要 40 s，分流後剩餘用量只省 25 s，因此不建 gate（算例）。原驗收只核對選路，未量整次演化加速。
- 2026-11-18，[Full-drive gate 的 CPU 競品估價](./2026-11-18-gate-direct-duration-price.md)，`adopted`。按 row duration 估算 CPU 競品，避免在不回本的列先建多 tone gate。新版係數在 25 ns 的 0.0891 與 100 ns 的 0.412 之間內插（價格設定）。Flat-top lab 留出案例未量，原收據未釘量測 commit。
- 2026-11-18，[Sampled multitone RWA 恢復准入](./2026-11-18-sampled-multitone-rwa-admission.md)，`adopted`。用逐列條件與同形群抽查控制誤差，只有預估回本才付探針成本。正例 wall 中位加速 3.149 倍（`9a09218c` → `28afb820f`，CPU 7 workers、d=24 的 `simultaneous_drive_retained24` 24 點、三組 ABBA）。
- 2026-11-16，[常數雙 tone secular guard](./2026-11-16-constant-multitone-guard-diagnostic.md)，`adopted`。補查被省略的耦合在長脈衝上的累積誤差。最後 admission 的累積量上限為 `1.476e-4` cycle（guard 門檻）。未找到可追溯的量測 commit。
- 2026-11-16，[Floquet 保留與價格重校](./2026-11-16-floquet-kept-price.md)，`adopted`。保留有配對收益的 Floquet 路徑，重校與直接積分比較的成本。16 態的 Floquet 每列估價係數固定為 0.008 s（價格設定）。保留研究與重校未各自釘量測 commit。
- 2026-11-15，[common_frame 收窄 coherent guard](./2026-11-15-common-frame-coherent-narrowing.md)，`adopted`。降低既有 coherent 門檻，讓超標列回到其他由價格選出的 family。Coherent 上限 0.03 → 0.0018（設定值）。未釘格點計時版本。
- 2026-11-15，[t6 移除 reusable channel](./2026-11-15-t6-channel-removal.md)，`adopted`。退役代表工作負載未選用的 channel 分支，保留 t6 direct。若日後遇到同一 key 有 20,000 個載波週期的高重用輸入，可重評 channel（合成用例，非實測）。原工作負載對照未釘量測 commit。
- 2026-11-14，[single_tone_rwa 保留 endpoint kick](./2026-11-14-single-tone-rwa-kick.md)，`adopted`。在兩端補一階 kick 並用 κ2 放行，保住較快的 RWA 路徑。κ2 上限 0.08（設定值）。原驗收數字未逐筆釘量測 commit。
- 2026-11-14，[暫停 sampled multitone admission](./2026-11-14-sampled-multitone-interim.md)，`paused`。暫停兩個 multitone 快路徑的 sampled 列，保留原 fallback 及 RWA 的 constant pairs。維持完整序列最終 density 最大絕對差 `2e-3` 的驗收門檻。原討論未提供可追溯的加速數字。
- 2026-11-14，[common_frame 步長成本歸 dispatch](./2026-11-14-common-frame-step-pricing.md)，`adopted`。移除 kernel 的 2 ns 經濟性限制，由 dispatch 按步數和建構攤提選路。移除 2 ns 步長下限（設定值）。
- 2026-11-14，[t6 的 direct 與 channel 分派](./2026-11-14-t6-direct-strategy.md)，`adopted`。當時按 acquire 的 key 重用次數在 channel 與旋轉座標 direct 積分間擇一。21 態的 channel 建構估價為 31 s（當時價格式）。原計時未釘各側量測 commit。
- 2026-10-18，[Two-tone Floquet](./2026-10-18-two-tone-floquet.md)，`adopted`。保留近共振 sidebands，短段另比較 direct 的成本。早期 frame builder 版為 9.214 s（`58ca8b968`，S9 五點、兩 workers，暖機後交錯三次的 acquire wall 中位數）。
- 2026-10-17，[Dyson 的輕量分類與事後檢查](./2026-10-17-ftf-dyson-guard-and-classification.md)，`adopted`。分類不預先跑整段演化；Dyson 的結果超過偏差門檻時記錄而不重算。當時設定 `|Δp| ≤ min(0.02, (1.5·budget)^(2/3))`。多輪驗收未逐筆釘量測 commit。
- 2026-10-17，[退役早期 reduced RWA](./2026-10-17-retire-reduced-rwa.md)，`adopted`。移除在當時 FTF 輸入上未增加覆蓋的 constant-tone 後援。舊分支要求 `r ≤ 0.2` 且省略 sidebands 的誤差界不超預算（舊 guard）。沒有可追溯的前後計時。

## 跨點重用與快取

- 2026-11-23，[放寬 RB 跨 depth 的前綴候選](./2026-11-23-rb-prefix-candidates.md)，`rejected`。允許更多共同邊界未顯示整輪收益。放寬及只留最長邊界都約 103.4 s，原規則 97.6 s（`8cd18afcd`，離線 plain RB depth `[0,12,24]`、兩臂，單 worker，每規則一次 `run_stage` 牆鐘，Numba 暖度不同）。
- 2026-11-23，[FTF sampled RWA 探針快取](./2026-11-23-ftf-rwa-probe-memo.md)，`adopted`。一次 preparation 內重用完全相同的 sampled RWA 探針結果，每次 evolve 仍獨立選路。基準的 full-drive DOP853 探針用 175.29 CPU s（`ffd2a0c6`，離線 `two_qubit/chevron` 四點、單 worker 的一次 warm Run，呼叫執行緒 CPU 總和）。這不是快取後的耗時。
- 2026-11-22，[ADC raw plan 在 acquire 內重用](./2026-11-22-adc-raw-plan-acquire-owner.md)，`adopted`。把確定性的 ADC raw plan 綁到 acquire session，換批或結束才退役。Raw materialize 343.737 → 159.082 s（`e602f0dc6` → `a4cd9f1bd`，同 fixture 的完整 `chain_power`，非排他單 worker、各一個 Run 的 materialize 累計）。
- 2026-11-21，[Projective 腔場 span 重用](./2026-11-21-projective-span-reuse.md)，`adopted`。由 ADC 一次取二階矩與各態端點場，完全相同的 span 才重用。獨立端點解從 29,921 次降到 0（`afbed1182` → `aef142668`，seed 123、`chain_length` 2.2 µs 首點、Engine ordinal 0–4 前綴的計數）。
- 2026-11-21，[MIST 腔場端點重用](./2026-11-21-mist-cavity-reuse.md)，`adopted`。擴充 ADC 的 span memo，讓未作用的 pulse 振幅不阻止端點場重用。候選完整 Run 為 163.3 s（`767c478b8`，排他 CPU 0–7、seed 123、101×101 點、1000 reps × 10 rounds 的 `mist/flux_dep`）。
- 2026-11-18，[成形脈衝的 fallback 共用](./2026-11-18-shaped-fallback-key-reuse.md)，`adopted`。先解析可共用表的 Sambe guard，讓 fallback 可跨 timeline 共建 period table。合成驗收要求 8 條 fallback 列只建 1 張表（測試條件，非計時）。未取得落地版配對耗時。
- 2026-11-16，[凍結場景跨 worktree 共用](./2026-11-16-frozen-scene-storage.md)，`adopted`。每筆凍結場景用 1 個 gzip，放在 repo 共用快取（存放設計）。Git 只留身份清單。舊 bundle 與 dirty-tree 計時未釘 commit。
- 2026-11-16，[Floquet 不重建被拒候選的 frame](./2026-11-16-floquet-refusal-cost.md)，`adopted`。把必拒條件提前，並在一次 evolve 內共用 exact-key frame。Legacy 未通過時，提前檢查 `spread*T > 14`（沿用既有拒收門檻）。兩組配對未各自釘量測 commit。
- 2026-10-18，[Dyson 輸入共用](./2026-10-18-dyson-input-reuse.md)，`partial`。完全相同的前綴只演化一次，後續共用收縮表。未採用的 sparse scan 只省 12.6%（`16b1dde43` 上的原型，R37 201 個 carriers、兩 workers，各版一次含 input capture 的完整 acquire wall）。
- 2026-10-18，[Shaped periods 與 gates](./2026-10-18-shaped-period-and-gates.md)，`partial`。重用 guard 的建表結果，只在窄掃頻組插值。原型觀測少 25.15%（`b0cb487f3` 上的原型，F4 窄頻 2001 點的第一組完整 acquire wall 對照，受其他程序負載干擾，不能判定加速）。
- 2026-10-16，[FTF 每點演化一次再抽樣](./2026-10-16-ftf-simulation-speedup.md)，`adopted`。以 runtime 分派與快取取代離線表和重算，將物理演化與 shot 抽樣分開。舊路徑單個 128 ns twotone 點為 2.9 s（`e7f32506c`，單點原型的改前基準，不是完整 sweep 或落地版）。
- 2026-10-15，[離線 propagator library](./2026-10-15-offline-propagator-library.md)，`partial`。把重複建表移到離線，執行期載入並插值。雙 amplitude 的四階插值用 25 個直接格點，六階用 49 個（建表設計）。早期原型未釘量測 commit。

## 參考積分

- 2026-11-22，[CPU DP5(4) 縮短每個區間的起始步長](./2026-11-22-dp54-interval-step-seed.md)，`adopted`。每個區間從更小的第一步開始，以減少重試，不改容差。0.803 → 0.616 s（`47fbcf5e8` 上的原型，heralded RB 的 400 ns 雙 tone 末段，排他 CPU 的單列 apply 中位數）。
- 2026-11-21，[Heralded RB 的 DP5(4) 列內重用與共同步長](./2026-11-21-rb-dp54-row-reuse.md)，`rejected`。列內重用和批量交換子沒有足夠收益。辨認出的重用項目樂觀估省 14.29%（`666eb48ac`，heralded RB 的 400 ns 雙 tone 末段，從單列 apply 分項耗時估計，非實作加速）。
- 2026-11-18，[短 dop853 row 改用 DP5(4)](./2026-11-18-short-dop853-dp54.md)，`adopted`。保留 family 契約，合格的短 row 在 CPU 和 OpenCL 走 DP5(4)。CPU wall 中位加速 1.4986 倍（`d45e87637` → `62844ad96`，d=24 的 `simultaneous_drive` 24 點、7 workers、三組排他 ABBA）。
- 2026-10-21，[Joint Fock CKP 的 RK4 與 cutoff](./2026-10-21-joint-fock-performance-study.md)，`partial`。每個 rf 共用準備產物。保守候選採固定 1 ns RK4 步長，總激發數 cutoff 取 pointer 自適應值再加一（原型設定）。成本為外推，研究量測未釘 commit。
- 2026-10-18，[Direct integration](./2026-10-18-direct-integration.md)，`partial`。保留自適應控制，合適的單 tone row 改用 interaction picture。原型少 31.50%（`b0cb487f3` 上的 IP 加 gain grouping 原型，F5 201 點、兩 workers 的一組完整 acquire wall 對照，非正式版）。

## 表示與合併

- 2026-11-24，[異形同載波的 shaped Sambe](./2026-11-24-same-carrier-shaped-sambe.md)，`paused`。保留各線包絡，在共同載波座標加總 Fourier block；兩場 guard 均拒收。既有 DP5(4) 0.592 CPU s（`fd160e255`，FAST-DRAG／cosine 雙線，8 態、100 ns，一次 Engine 演化）。
- 2026-11-23，[同頻多線摺成單 tone](./2026-11-23-same-carrier-merge.md)，`adopted`。同載波同形線合成一對 I/Q 算符，交給單 tone family。Plain RB 16.401 → 0.892 CPU s（`2b8f628c3` 對 `0d851985`，depth `[0,4,8]`、seed 701、sim seed 123、2 reps、1 round，單 worker 暖機後各一次）。
- 2026-11-22，[FTF 真空續演化與 Markov chain](./2026-11-22-ftf-projective-markov.md)，`partial`。投影後清零腔場，讓同點、同層、同 DC 態只續演化一次。獨立同模型參考通過 41／75 個 key（`e602f0dc6`，兩個 floor，兩個縮樣讀出掃描及 heralded conditional phase 的首讀）。
- 2026-11-18，[FTF 的 Engine 內子空間約化](./2026-11-18-ftf-subspace-reduction.md)，`rejected`。比較低能截斷、逐點選態與誤差界，尋找便宜又可靠的自動約化。24 態對 32 態的完整 density 最大絕對元素差，15／15 點均 `≤1e-3`（非整體精度目標 `2e-3`；`c04fb3823` 上的研究原型，固定 DC、F1 單一常數 tone 的三組五點掃描）。
- 2026-11-18，[FTF acquire 動態維度](./2026-11-18-ftf-dynamic-dimension.md)，`adopted`。用最強一點做上限 pilot，只有預估節省足以抵驗算時才試小維度。至少 8 條 timeline 才納入（准入條件，另要求只有終點讀出、沒有早期 cavity drive）。原討論未對各輪釘量測 commit。
- 2026-11-17，[Single／six projective Markov chain](./2026-11-17-projection-markov-chain-readout.md)，`adopted`。以逐態轉移取代 history 展開，讓輸出不再隨 read depth 組合增長。Producer 70.0 s（`60cc37798`，單次 `ChainAmpRabiExp` 的 51 點、16 次 read event、21 態，8 GiB／無 swap；只包 producer，不含整次 acquire）。
- 2026-10-28，[FTF 固定 DC flux ramp 截斷](./2026-10-28-ftf-fixed-dc-flux-benchmark.md)，`investigating`。試較小的固定 DC 子空間，並把長 hold 以常數段表示，避免依 hold 長度加密波形。最長 hold 仍為 11 nodes（`84e9a420a` 上的探針，合成 FTF F1 reset、M8/local4、8 ns ramp、hold 1866.200 µs 的單點 timeline）。
- 2026-10-21，[CKP reduced closure 取捨](./2026-10-21-ckp-reduced-closure-study.md)，`partial`。比較哪些腔場歷史可壓進有限維 closure，避免完整 Lindblad 的成本。Displaced-Fock 的候選之一把總激發數截在 `N=2`（模型設定）。研究量測未釘 commit。
- 2026-10-18，[Constant-period builders](./2026-10-18-constant-period-builders.md)，`partial`。按 key 選 period builder，只沿 gain 軸續算。Carrier continuation 零命中（`b0cb487f3` 上的原型，L7strong 的 1103 次預測全數未通過原檢查）。

## 平行化與排程

- 2026-11-23，[Chevron 先減少 caller 準備成本，再評估程序池](./2026-11-23-chevron-parallelism-order.md)，`paused`。直接換 `loky` 無法序列化現有工作。既有執行緒由 1 改 7 workers 快 1.28 倍（`ffd2a0c6c`，離線 2×2 gg herald chevron、非排他 4 CPU、各一次完整 Run 牆鐘）。
- 2026-11-21，[FTF 同點同 span 的投影續算合批](./2026-11-21-projective-continuation-batches.md)，`adopted`。同點同 span 的就緒 histories 一起送到 Engine，保留各列初態。固定兩態無驅動序列的配對重播設計從 8 次 evolve 改為 3 次，批寬 2、2、4（呼叫分組，非 ABBA 觀測值）。原計時兩側 commit 未釘，不列加速。
- 2026-11-18，[dop853 的 OpenCL GPU 後端](./2026-11-18-ftf-gpu-dop853-backend.md)，`adopted`。把合格的獨立 row 批次送到 FP64 GPU，沒有可用裝置時保留 CPU。CPU／auto 中位 wall 比 5.677 倍（`544af11bf`，Intel Arc、d=32 的 `simultaneous_drive` 24 點、7 workers、三組獨占 ABBA 的整次 evolve）。
- 2026-11-17，[DOP853 路線比較](./2026-11-17-dop853-speed-route-summary.md)，`partial`。配對比較同批次的 worker 寬度，另查載波積分、前導共用與 GPU 是否值得續做。8 workers 的 wall 加速中位數為 4.002307 倍（`2514d94be`，pinned DOP853 的 64 點 Gaussian 單段，對 1 worker、8 輪 warm_clear_history 的 evolve 配對）。
- 2026-10-18，[Planner 的 evidence admission](./2026-10-18-planner-evidence-and-overhead.md)，`rejected`。按完整 acquire 評估候選路徑，未通過效能驗收。校準域為三能階合成模型的 20 個 fit 與 16 個 held-out 案例（驗收設計）。候選版的確切量測 commit 不明。

## 環境與讀出

- 2026-11-22，[FTF 樣板提高 projective floor](./2026-11-22-ftf-projection-weight-floor.md)，`adopted`。提高新工作副本的 floor，減少低機率來源態的續演化。最大完整 density 差 6.921e-6（`cc6b252c0`，六列 FTF 輸入、同 seed、Markov 模型的 floor `1e-5` 對 `0`）。
- 2026-11-18，[移除 private replay](./2026-11-18-remove-private-replay.md)，`adopted`。移除逐位重算及舊 archive 相容要求。為逐位重算，開發時曾把 pilot 最少 timeline 數從 8 提到 16（舊設定）。沒有單獨量測移除的 runtime 收益。
- 2026-10-18，[Flat integrated readout 離散和](./2026-10-18-readout-discrete-sums.md)，`adopted`。對 ADC 樣本的條件場直接求幾何級數，省掉產生整條 trace 後再加總。研究原型要求兩個 eigenbasis 的 condition number 均 `<1e4`（研究 guard）。可取得基準樹，未取得原型的量測 commit。
