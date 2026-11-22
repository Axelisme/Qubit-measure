---
status: accepted
---

# 0072 — 模組定位與依賴邊界

## 脈絡

原先 `zcu_tools` 的 15 個頂層 package 只有 7 條 import-linter contract。其他邊界靠文字約束，agent 容易把既有耦合當成先例，也不會主動寫 contract。

例如 `experiment/base.py` 的 `PersistableExperiment` 呼叫 `datafile`，卻自行處理 Labber 版面、讀回驗證、通道型別與 comment 編碼。當時能力模組也出現兩個循環：`resources → program → simulate → resources` 和 `resources → program → analysis → simulate → resources`。

使用者於 2026-10-04 的討論決定 package 定位。本篇規定各 package 的責任與依賴方向，由 `.importlinter` 檢查。尚未搬移或仍以 `ignore_imports` 記帳的項目列於〈債務〉。

## 決策

### 模組定位

**規則。** 新程式碼按責任放置。所有其他類別都不合，才放 `utils`。

| package | 負責 | 不負責 |
|---|---|---|
| `datafile` | Labber、`data.h5`、streaming、grouped 等量測資料檔；版面、讀回 schema 驗證、通道型別與 metadata 編碼 | 存檔位置與批次（[[0063]]）；實驗語意 |
| `device` | pyvisa 儀器定義與控制介面 | 實驗流程 |
| `program` | ZCU216／QICK 控制、模擬 SoC binding、板子 profile 到 QickConfig 的轉換，以及由檔案或記憶體建立 arb envelope | 參數及資產保存；不在 lib 存硬體時脈、channel 型別或數量 |
| `analysis` | 分析工具及模型自己的公開 renderer，使用與客製化由 caller 決定 | 檔案讀寫；實驗的圖形政策 |
| `plotting` | 具名 Figure、owner 更新機制、共用診斷圖 | 分析計算；lab 的 live 工具與格式 |
| `simulate` | 數值模擬、通用數值輸入與參考解 codec | 讀取量測資料容器、裝置佈局或使用者環境參數；模型數值由 caller 提供 |
| `simulators` | FTF、single_fluxonium、six_fluxonium 的具體物理模型與 environment | 通用演化、ADC、bath、元件 Hamiltonian 等，這些留在 `simulate` |
| `resources` | 實驗領域資料容器與 YAML、JSON、CSV I/O；waveform 資產命名、formula 語意與 NPZ 實體 I/O | 量測資料檔；program 執行機制；重複實作 waveform codec |
| `core` | 共用無狀態型別與格式，例如 `core.waveform_format` 的樣本驗證、雜湊、NPZ 及 opaque JSON recipe codec | 檔案 I/O、formula 語意、GUI、資源 scope 或硬體 |
| `cfg_model` | 配置 schema 與注入依賴契約；仍在原位置，見[共享核心](#共享核心-core-與-cfg_model) | 執行行為與 I/O |
| `utils` | 無法歸入其他類別的工具 | 已有 owner 的能力 |
| `progress_bar`、`qick_remote` | 進度列及 QICK 遠端能力 | — |
| `experiment` | 實驗介面、AxesSpec、RunRecord、RunContext、runtime 與註冊機制 | 檔案格式；具體實驗 |
| `gui` | 各 GUI app 的框架 | 具體實驗 adapter |
| `notebook` | Notebook 特化工具 | 通用分析；具體實驗 |
| `mcp` | 各 MCP 的框架 | 具體 recipe |
| `apps`、`mcp_servers`（repo 根目錄） | GUI 與 MCP 組合根，注入 lab 定義與 GUI module 名稱 | framework 與 lab 的可重用實作 |
| `zcu_lab`（repo 根目錄） | 具體實驗、recipes 與 GUI、Notebook 附件 | 離線遷移的對照與宣告，這些放 `migrations/` |

**例子。** 需要從 caller 的板子 profile mapping 建 QickConfig，用 `program` 的 `build_soccfg`。如果要查 FTF 或 single_fluxonium 某次演化的 Engine 輸入、逐列 routes、seed、每輪 DC、model identity 或當時有效設定，開 `numerics.debug_capture` 後讀 manifest。見[模擬器的 debug capture](../../lib/zcu_tools/simulators/README.md#debug-capture)。

**邊界。** 需要更多除錯資訊就補 manifest，不為了重播重建舊數值路徑。設定與機制的分工見[下一節](#框架機制與模型參數)。

**依據。** Package 定位出自 2026-10-04 的使用者討論，原始討論未入庫。2026-11-18，使用者：「你需要的资讯应该本身就在 dump 的结果里了。如果你会需要 replace 多余的资讯、多余的算法，那就是你 dump 的资料不够多嘛。」見[移除 private replay](../discussions/2026-11-18-remove-private-replay.md)。

### 框架機制與模型參數

**規則。** 框架提供機制和最簡單的範例，不硬編模擬參數。Caller 以獨立 YAML 提供板子 profile，`zcu_tools.simulators` 擁有具體物理模型。

**例子。** `white_noise` 是最簡單的框架範例；編譯所需的板子設定從[呼叫端 YAML](../../lib/zcu_tools/program/v2/README.md#呼叫端-yaml-編譯設定)取得。

**邊界。** 具體模型不為了提供範例搬入框架。配置值不因 lib 提供讀取或編譯機制而變成 lib 的常數。

**依據。** 2026-11-01，使用者：「框架應該只有機制本身以及最簡單的範例，比如white_noise，絕對不可以hardcode參數」。回答板子 profile 的來源「b，模擬器就是要能設定模擬參數」，回答具體物理模型的位置「A」。原裁決未附入庫討論，見 Git `53a726c3d`。

### 模擬器是封閉的子模塊，模型由使用方聲明

**規則。** `simulate` 不 import 其他 `zcu_tools` package，其他 package 可以依賴它。前端與 environment 共用的 acquisition 邊界型別歸 `simulate.boundary`。`simulate` 提供聲明模型的介面，使用方負責裝置組合。模擬器量測工具與測試只依賴模擬器並接收模型聲明；量真實模型時，由使用方的入口注入，工具不 import 具體模型。

**例子。** Caller 以 `ModelSource` 注入 `build: Callable[[FluxLevel], SecularModel]`。`SingleFluxoniumModel`、transmon 和 resonator 模態 builder 只接物理參數，留在 `simulate`；FTF 耦合與 resonator 配置、six 路由由 `simulators` 定義。邊界資料與 Numba 快取細節見 [Acquisition boundary](../../lib/zcu_tools/simulate/README.md#acquisition-boundary) 和 [Numba machine-code cache](../../lib/zcu_tools/simulate/README.md#numba-machine-code-cache)。`simulate` 自行解析快取根目錄，規則與 `utils.cache.user_cache_root` 相同，封閉不改快取位置。

**邊界。** 測試用的兩能階等合成模型可隨 `simulate` 的測試放置。三個 environment 不共用基底或 acquire 流程，只共用中性的 telemetry writer、欄位型別與 evolution 錄製 buffer。FTF 的腔記憶和 six 的 scatter 不做成可選 hook，見 [ADR-0078〈模擬器帶通用調試介面〉](0078-evolution-module-boundary.md#模擬器帶通用調試介面)。C19 仍禁止 `core` import `simulate`，舊 `core.simulation` 不留 re-export；`simulate-closed` 連間接 import 也檢查，不設 `ignore_imports`。新增頂層 package 時同步加進清單。本節不改 `simulators` 的定位或[工具與測試不讀使用者環境參數的規則](#使用者空間的依賴原則)。整包獨立仍是[草稿方向](draft/simulator-standalone-package.md)。

**依據。** 2026-11-15，使用者：「先告訴你，我預計將模擬器功能獨立出zcu_tools成獨立的package，後者依賴前者，目前作為封閉子模塊可以」。對模型位置答「C，模擬器應該提供一個方法讓使用方透過繼承或是組合或是依賴注入的方式讓使用方聲明模型」，對 `SingleFluxoniumModel` 的位置答「1A」，對原則與真實 FTF 量測的推論答「A」。見[模擬器封閉成子模塊](../discussions/2026-11-15-simulator-closed-submodule.md)與[搬遷順序](../discussions/2026-11-17-simulate-closure-expand-order.md)。

### 真實 acquire 與模擬邊界

**規則。** 真實硬體 acquire 不執行模擬模組，包括 `simulate` 及 `program.v2.simulation` 的排程器。共用估時留在中性的 `program` 工具。模擬 caller 明示 generator 必須綁到 charge drive line 時，由知道實體線路的 environment 驗證。Program 只將需求傳進 `AcquireRequest.charge_requirements`；`simulators.replay` 保存同一種 boundary record，不重做驗證。

**例子。** STIRAP 以 `ChargeDriveRequirement` 傳入診斷名稱和 generator index。名稱不是 role registry key，caller 不查環境設定或判斷線路成員。Single-fluxonium 每次 acquire 在 waveform 建構、preparation reuse 與 evolution 之前用 `drive_lines` 驗證；已綁 flux 或 readout 不算 charge，也不改其他合法 fast-flux 路由。見 [program 的模擬入口](../../lib/zcu_tools/program/v2/simulation/README.md)與 [single environment](../../lib/zcu_tools/simulators/single_fluxonium/environment/README.md)。

**邊界。** 硬體入口取走 keyword，不轉送 QICK、不建模擬 request、不做物理驗證；operator 核對實機線路。`white_noise` 沒有物理線路，接受但不驗證。Six 與 FTF 等不支援此需求的物理環境在 evolution 前拒絕非空需求，空需求不改流程。不清空需求、不從 waveform 猜角色、不擴張 `QubitWiringSchema`，也不另加 Protocol 或 `RunContext` capability。Program、experiment、simulators 與 lab 可 import `simulate.boundary` 的 record；載入型別不執行模擬，`simulate` 初始化只載入 numpy。C8、C9、C12、`simulate-closed` 與 `program-simulation-entry` 檢查依賴，program class 只能經入口使用模擬 frontend、timeline 與 adapter。真實 acquire 是否執行模擬仍要審閱執行路徑。

**依據。** 2026-10-26，使用者：「我不太希望真实 acquire 还会走模拟的模块。」2026-10-30，advisor 依授權「可以按需排task，我授權你代理」代裁：「驗證 owner 是 single_fluxonium 環境，在既有拒絕 unbound generator 的 preflight 位置一起檢查，早於任何 evolution。」未見使用者覆核這項代裁的紀錄。原裁決未附入庫討論，見 Git `d657120b9`。

### Mux PFB channel 查詢與 carrier selection 的 owner

**規則。** Program 擁有 QICK PFB channel assignment 的查詢；FTF 只做 event／carrier selection，不組 QICK register 結構或重做 folding 算式。

**例子。** `program.v2.mocksoc.pfb_channel` 用 caller 提供的 QickConfig 呼叫 QICK 公開的 `calc_ro_freq` 並取 `pfb_ch`。見 [program 的查詢契約](../../lib/zcu_tools/program/v2/README.md#呼叫端-yaml-編譯設定)。

**邊界。** Experiment 沿 C8、C9 依賴 program 取得查詢結果，不新增或放寬 import contract。

**依據。** 2026-10-31，使用者對 mux 濾波答「Mux的可以」。Advisor 依授權「1代理模式就都給你決定，我事後看就好」代裁 owner 分工，未見使用者覆核這項技術細節的紀錄。原裁決未附入庫討論，見 Git `5c15e701d`。

### Manifold 衰減模與分析診斷的 owner

**規則。** `simulate.fluxonium.coherence` 組 directional rate matrix 並計算所有衰減模。`notebook.analysis.t1_curve` 選主模、算次模佔比並標記圖表。

**例子。** Manifold forward 每個 flux 點回傳全部 `Mode`，以 signed amplitude 由大到小排列；`t1_curve` 取振幅最大的模作 model T1。局部型別、結果與失敗契約見 [coherence README](../../lib/zcu_tools/simulate/fluxonium/coherence/README.md) 和 [t1_curve README](../../lib/zcu_tools/notebook/analysis/t1_curve/README.md)。兩種 T1 的物理定義與共同單位見 [[0071]]。

**邊界。** Simulate 不選主模、不回傳 secondary fraction、不套非單指數標記門檻。C8 允許 notebook → simulate，C9 檢查無循環，不新增或放寬 contract。

**依據。** 2026-10-28，使用者：「我覺得工具就應該保持單純」。Advisor 復述為 simulate 回傳所有模、t1_curve 選主模並做診斷，使用者答「對」。原裁決未附入庫討論，見 Git `6a9341bc2`。

### 模擬損耗漂移的數值、設定與 truth owner

**規則。** `simulate.drift` 擁有純 log-OU 數值。`resources.single_fluxonium.schema` 擁有 single、six、FTF 共用的 `DriftParameter` 和 `CoherenceDrift`。Experiment 擁有設定編排、acquisition 開始的取值凍結與 truth log I/O。每個環境自己組 effective settings 和 loss streams。

**例子。** FTF 透過 `experiment.single_fluxonium.drift` 的 `LossRow` 和 `write_truth` 寫三個環境共用的 truth JSONL。Coupler 沒有的 q_ind／xqp 記為 null。FTF 不複製格式，也不建 single settings adapter。FTF 與 single-fluxonium 啟用 `numerics.debug_capture` 時，manifest 的 `metadata.effective_settings` 記錄觀察到的 round model identity 當時的設定，包括取樣後損耗；發布用對應的 round snapshot。

**邊界。** `LogOU` 只接 mean、sigma、tau_s 與 RNG，用 caller 提供的時間算精確轉移，不讀 YAML、clock 或檔案。見[純數值漂移](../../lib/zcu_tools/simulate/README.md#numeric-parameter-drift)。Truth 是模擬 acquisition 診斷，不進 Run record、GUI、MCP 或 Labber metadata。C8、C9 下仍是 experiment → simulate／resources，resources 不反向依賴 experiment。物理前提見 [[0078]]。

**依據。** 2026-10-27，使用者：「另外我想引入跨acquire的T1/T2漂移，這樣overnight量測才有意義」。Advisor 依本篇模組定位與使用者授權「B你可以自己開task，不用我看過」，於 2026-10-27 與 2026-10-28 代裁 owner 分工。未見使用者覆核分工的紀錄；2026-10-28 的「寫」允許記錄，未單獨追認細節。原裁決未附入庫討論，見 Git `257e8cc3a`。

### 實驗計算與 GUI 接線

**規則。** 各實驗在自己的 `core.py` 取數、分析與畫圖。Core 擁有擬合、逐 tone 計算、失敗隔離、結果型別和 `run` 的即時圖。`gui.py` 接線、解析來源、決定 writeback 與 operator guide。

Lab 回傳給 measure GUI 的 analysis 摘要最多隨掃描點數 O(log N)。數值 helper 可回完整結果，core 用它畫圖。摘要型別由數值 helper 的 owner 定義，GUI 不另定型別、不重算或重跑 fitter。

**例子。** Mux onetone core 只收每 tone 的 edelay 種子，回傳擬合與單 tone outcomes；adapter 選手動值、capture prior 或全域搜尋，記錄來源、解析 route prior 並決定 writeback。Mux flux 各 leaf 自己建 program 並跑 flux 迴圈，共用工具只算數值、設定 flux 和組 Result。Mux T1／T2echo 回傳本次模型擬合，比較替代壽命模型與疊圖由 `zcu_lab/workflows/common` 擁有。Local flux reset 用 lab 的範本組裝 `reset/pulse`，由 [gain](../../zcu_lab/v2/twotone/reset/flux/gain/guide.md) 和 [length](../../zcu_lab/v2/twotone/reset/flux/length/guide.md) 校準，重用 reset check，不新增框架 reset 型別或專用 check。

**邊界。** 不呼叫其他實驗的 `run` 或 `analyze`，也不組假 source 或用旗標隱藏各 leaf 的取數。含 `core.py` 的實驗目錄不 import 其他實驗或其他家族的家族層模組；自己家族的 helper、`support` 及家族 `__init__.py` 匯出子實驗不受此限。Cfg、validator、分析選項、結果與摘要、本地小訊號轉換、GUI 選項及 review 文字留在使用它們的實驗或數值 owner；實驗只宣告用到的欄位，放在使用它的檔案。Cfg 不繼承其他實驗，原 validator 除另有裁決須保留。

共用資料分析進 `zcu_tools.analysis`；認得 lab 型別的放家族私有模組或跨家族的 `zcu_lab/v2/support`，單一家族的序列產生與查表留家族層。Helper 換 owner 不留 re-export。Lab／support 不替多個實驗建分析圖；`onetone/freq`、`mux/onetone` 用 `support.resonator.fit_resonance` 的數值結果，在各自 Figure 決定何時畫、是否用 lib 模型的公開 renderer，不複製 renderer 或呼叫私有方法。Mux GE 各自逐 tone 用 `analysis.singleshot` 的 GE confusion 工具並畫圖。分類、population correction 與 confusion 數值同屬該 owner，只收明填選項；各 lab 實驗從自己的分析結果組 `GECalibration`，不傳 lab cfg；預設、存檔、GUI 文字及圖也留在實驗。介面見 [analysis README](../../lib/zcu_tools/analysis/README.md#子模組)與 docstring。

GUI 為 writeback 檢查有限值或整理摘要不算分析。發布前 `summarize_analysis` 拒絕任一攤平後超過 64 元素的 ndarray、list 或 tuple，raise `ValueError`，不截斷。小於 64 的 O(N) 資料仍不是摘要。隨掃描點數或 shot 數成長的陣列只在 `analyze` 內畫圖，不回傳。`RabiJointFitSummary`、`Centroid3Summary` 是小摘要。C8、C9、C15、C16 保護依賴方向，計算與接線的分工用直接審閱，不新增靜態內容測試。

**依據。**
- 2026-10-24，使用者：「怎麼會有分析寫在gui ,這是違規吧」「分析修正可以做」。
- 2026-10-30，對 T1／T2echo workflow 比較答「18A」；對 edelay 指出「這不對勁，聽起來像是adapter的邏輯漏進core」，答「A」。對共用分析圖說「不能獨立重複分析嗎？可以共用分析工具，但畫圖就有點超過了，不利於客製化」。對 flux reset 答「1A 2同意 3感覺復用reset/check就好？倒是reset/可以加一個flux_reset，4同意」，並對 lab 組裝與校準答「確認」。
- 2026-11-02，使用者：「為什麼會把很大的數據用json.dumps？這不合理」，裁定「summarize_analysis可以檢查結果的大小，太大就報錯。summarize請不要放隨著掃描參數成O(N)以上的結果。最多O(logN)」，對 64 元素與修改 core 答「64可以，CPMG選a，core」。同日對獨立取數答「a，我想討論的是initial_values與Form.default_input的定位問題」。
- 2026-11-12，使用者對各實驗獨立分工答「確認」，說「實驗間重複算是偶然耦合而不是實際耦合。」2026-11-12，對數值與繪圖分離及公開 renderer 例外裁定「接受」。2026-11-12，對 GE confusion 的 owner 與介面答「可以」。
- 原裁決未附入庫討論；見 Git `2df4196e2`、`3a75e54ca`、`e94d0f30f`、`df5d654dd` 與 `a95315c32`。

### Two-tone 廣掃的選峰與窗口

**規則。** `twotone/freq` 暴露 Model type、Expected center 和 Min SNR。`analysis.fitting.spectrum.fit_spectrum` 以明填 `local_peak=True` 負責數值選峰與窗口。Lab 留下選項、IQ 投影與圖上的實際窗口標示，數值工具不讀 lab options。

**例子。** 同一廣掃有強 transition 與一個高度不到它 0.2 的弱 transition 時，可用 Expected center 選附近最高的局部極大值，而不因全譜 range 比例拒絕。中心空白時選最顯著峰。Min SNR 仍可拒絕弱峰；失敗回報候選數與門檻，不改擬合遠處強峰。預設 SNR 5 見 frequency guide 的[序列與設定](../../zcu_lab/v2/twotone/freq/guide.md#sequence-and-key-settings)；左右各六個半高寬、中點裁切但至少取最近六點的契約見[頻譜判讀](../../zcu_lab/v2/twotone/freq/guide.md#read-the-spectrum)。

**邊界。** 只改 `twotone/freq`。`SpectrumFitOptions` 保留欄位，`twotone/chain` 仍用最近候選與中點窗口；Notebook fluxdep 與 workflow calibration 不改。其他 caller 自行明選門檻或窗口，不繼承 freq 的 GUI 選項。C8、C9、C15、C16 下仍是 lab → analysis。

**依據。** 2026-11-16，使用者：「你twotone/freq的算法不好，對於廣掃，多transition peaks是很常見的，gui控制應該只需要model type/expected center兩個就好。」並補充「取6個半高寬。其他門檻，如果預設可能會導致找不到峰，那要考慮曝露。」對保留 Min SNR、移除相對 prominence、其餘兩門檻固定為零答「A」。依使用者要求未另建討論快照，見 Git `e827dbecc`。2026-11-18，使用者過目說「我大概看了一遍，基本上是没问题」，但說明這類規則「并不算是裁决，而算是建议」。此實作是 agent 的選擇，只作參考先例，見 Git `1f4b16aa0`。

### 即時圖工具與機制

**規則。** Lib 擁有 Plots、FigureCollection、PlotHost、owner 更新與 lifetime。已顯示的圖不論即時或一次性改動，都經 `@plots.on_plot_thread`。Lab 共用支援套件的直接層擁有 live helpers、typed handles 與線圖、熱圖、非均勻熱圖、scatter 格式，不另建 plotting 子套件。實驗作者自己排 subplots、修改格式或擴充工具。

**例子。** `live_line(ax)` 掛在指定 axes，作者在 decorator 本體回傳 `line.set(xs, signals)` 的 changed artists。Show 前只修改 artists，不 refresh 或載入 ipympl。後續 `fig.show()` 或 `finish(present=True)` 呈現最新圖。Lifetime、deepcopy、artist 歸屬、refresh 及失敗行為見[原生 matplotlib 更新](../../lib/zcu_tools/plotting/README.md#原生-matplotlib-更新)；其他 live 工具和資料格式見 [lab 的即時圖工具](../../zcu_lab/v2/README.md#即時圖工具)。

**邊界。** Lab helper 在 caller thread 更新，不建圖、不加標籤或標題、不 refresh、不 present，也不綁 Plots lifetime。Helper 與 decorator 不登記 axes 或檢查佔用；共用 axes 由作者協調 artists、autoscale 和色階。混合回傳只 refresh 已呈現的圖；外來圖即使未呈現也拒絕。Notebook／Qt host 的 refresh 前提不變。Fluxdep 的非均勻底圖由其私有 renderer 處理，不依賴 lab 工具；singleshot 專屬分類與 joint-fit 診斷留在 lab，數值 owner 不搬動。獨立 liveplot plotter／backend、reference marker、錄影以及 LiveUpdate、字串圖名 draw 和舊 helper 均不再使用。依賴仍為 lab → plotting，由 C8、C15、C16 保護。

**依據。** 2026-10-24，使用者：「框架與機制留在lib可以，live_line的定位應該是工具，可按需求擴充，但放在lib裡很難改。以及liveplot模塊應該可以考慮刪除，我以為plotting已經取代他了。」2026-10-30，使用者：「畫圖採用建議，但名字叫live不太好懂」、「@plots.on_plot_thread好了，隱藏執行線程的機制資訊可能在某些時候造成難以發現的bug」和「我也傾向後者，live_line只是本地執行緒的」。2026-11-01，全實驗完成 owner 更新和 lab helper 遷移。原裁決未附入庫討論，見 Git `8ba70ea3b`、`0a56d4bac` 與 `f44a7c8e9`。

### 分層

**規則。** 五層由下到上排列。上層可依賴下層，下層不知上層。

1. 能力模組：`datafile`、`device`、`program`、`analysis`、`plotting`、`simulate`、`resources`、`core`、`cfg_model`、`utils`、`progress_bar`、`qick_remote`。
2. 實驗框架：`experiment`。
3. 前端框架：`gui`、`notebook`、`mcp`。
4. 使用者空間：`zcu_lab`、`scripts`，內容與結構由使用者決定。
5. 組合根：GUI 在 `apps/`，agent MCP 在 `mcp_servers/`。兩者同層、不互相 import，可依賴 `zcu_tools`，對 lab 只直接 import `zcu_lab.assembly`。MCP 以注入的 module 字串啟動 GUI，不 import GUI 組合根。

**例子。** `scripts/transfer_results.py` 是獨立使用者入口，只把 checkout 的本地結果根目錄交給 `utils.result_transfer`。Rclone 外部傳輸不屬於量測資料檔格式或實驗容器；可重用實作依[模組定位](#模組定位)放在 lib，不在 `scripts/` 下建其他腳本會依賴的套件。C21 禁止 lib 反向依賴 scripts。

**邊界。** `zcu_tools` 不 import `zcu_lab`；框架也不 import `simulators`，由 `framework-no-simulators` 檢查。`tools/` 對 lab 也只直接 import assembly，固定 Pareto 場景不 import lab。`migrations/` 可直接 import 舊資料所屬的實驗與元件。Tests 不依賴 `zcu_lab`、`scripts`、實驗流程或預設值；`tests/scripts/` 既有的 `main` import 是債務。框架測試用最小合成 workflow／adapter，框架 UI 用 ComponentDriver。Review 檢查 import，必要時 grep，不暫移 lab 跑 pytest、不加 `ZCU_LAB` 環境變數或 import hook。

Lab 定義改動（含畫面）以 `experiment.probe` 分別驗 `run_stage`、`save_stage`、`analysis_stage`，各回具名結果。Run 和 analysis 回 PNG bytes，不寫磁碟、不串 e2e；save 寫 native `data.h5` 並 restore。三階段無 CLI。Agent 用 `synthesize` 與回傳 ArrayLike 的模型合成輸入，或用 `sample_iq_shots` 合成 raw IQ。Agent 看圖並記下判讀。Probe 與結果留 task scratch，不入庫為 durable 測試。API 與完成條件見 [offline experiment probe](../../lib/zcu_tools/experiment/README.md#offline-experiment-probe)及[測試文件](../../tests/README.md#ui-驗證)。

**依據。**
- 2026-10-12，使用者：「我給你一個判斷方法，如果我刪除zcu_lab測試會報錯，那測試就是寫錯了。」2026-10-16，改定查核方式：「算了，我認為不用真的測試ZCU_LAB移開後會不會出現錯誤。這中東西直接看有沒有import就好，偷偷做的在review也會被抓到，感覺是浪費時間測試」。Tests 不依賴 lab 的原則不變。原裁決未附入庫討論，見 Git `f63e7b3b4`、`1db0dd51f`。
- 2026-10-19，使用者：「不要把依賴放在scripts, 後者的每個檔案都該被理解為獨立腳本入口，需要用到的內容可以放lib裡」。原裁決未附入庫討論，見 Git `87b098d34`。2026-10-25，兩組合根的分層與注入裁決答「乙」，見 [[0068]]。
- 2026-10-30，使用者：「關於測試實驗定義的方式，用ui跑e2e有點太重量級了，我認為可以修改規則，透過建立context的方式測試run與analysis」，選「B」，即改驗收規則並提供共用 probe，各實驗只需提供實驗與 cfg。Durable lab 測試不在裁決範圍。2026-11-01，使用者：「run跟analysis本來就該分開測試，不要有e2e的測試」，stage-tests 見 Git `2f5e933c3`。原裁決未附入庫討論；[probe snapshots 討論](../discussions/2026-11-16-probe-snapshots.md)記錄了後續工具使用的邊界。

### 使用者空間的依賴原則

**規則。** GUI、MCP 組合根和開發工具需要 lab 宣告時，只直接 import `zcu_lab.assembly`，不重組其內部定義。固定 optimizer benchmark 的核心 pulse 序列與每個場景的模型值由工具持有，不 import lab 或讀可變的 lab YAML。入庫且持續使用的模擬量測工具與測試不讀 `zcu_lab/simulation_envs/`；一次性研究可讀已提交的環境參數。

**例子。** `apps/measure.py` 用 `lab_definitions()`，不 import `zcu_lab.v2.twotone.freq.core`，lab 可在入口後調整實驗、元件、recipe 與 workflow 的佈局。固定量表場景可重複核心序列以便直接閱讀，lab 改序列不會自動更新量表。量表有公開 program 編譯的場景、直接用 `SingleFluxoniumModel` 的單段場景，以及用公開 FTF 模型和多段 `TimelineBatch` 的 gate 場景。它們共用 producer CLI；固定設定、builder、`numeric_builder`、資源生命週期及路徑見 [Tools README](../../tools/README.md#fixed-scenarios-companions-and-reference-cache) 和 [Freeze numeric scenes](../../tools/README.md#freeze-numeric-scenes)。

**邊界。** `migrations/` 可直接識別舊實驗格式；若改成不同 lab 共用的框架，就改用 assembly。這不保證刪除 lab 後現有組合根仍可啟動，也不新增位置字串或 Python entry point 註冊。需要真實 lab 宣告的品質檢查仍走 assembly，不複製 lab。

凍結數值不入庫，各 worktree 共用 repo 快取，每筆 record 一檔。Git 只用 synthetic／real 兩份 index 釘住名稱和 hash，身份不含存放位置。Codec 由 `simulate.evolution` 擁有，只接顯式路徑，負責清單及單筆檔讀寫、hash 核對與原子發布；tools 解析 Git 位置和更新清單。數值不含 solver 或裝置 builder；producer 自持 `Scene`，每筆有入庫 builder，不依賴 tests 的案例。消費者和 tests 側參考解 producer 缺凍結檔就列 producer 指令並失敗，不 import producer 或 program 就地重建。重建不符釘住身份（包括 native thread 差異影響浮點末位）就停；必須明確更新清單並重算受影響的參考解。Report 用每筆的 `name`、`physical_sha256`、`oracle_sha256` 和 `metadata_sha256` 識別。參考解由 tests 經檔案交接，見 [[0078]] 與[開發工具的數值交接](#開發工具的數值交接)。

**依據。**
- 2026-11-12，使用者：「我其實不樂意看到tool依賴zcu_lab，我認為你要測試，那就只複製最核心的pulse序列來模擬啊，又不需要整個experiment」。另說：「你這個工具的目的是對優化器做benchmark吧？那場景自然是挑選過的」、「而且這樣tool也能提高可讀性，我寧可重複也要提高可讀性」。
- 2026-11-14，使用者：「其实应该是 B， 因为我对这个 repo 的期望是，我把 [zcu_lab] 删掉，也可以正常运作。而 [zcu_lab] 是使用者自由的、控制的、写的、安排结构的模块。」並澄清：「我的意思是 ，Script 的定位与 [zcu_lab] 比较像，而 Tool 的定位是用于开发。我不确定该不该让它依赖 Lab。MCP 需要发现、实验，而也确实是需要一种正规的方法来宣告你有哪些实验。我不确定要怎么好的结合两者，让 MCP 尽量不依赖[zcu_lab]。」對註冊取捨說：「但是呢，我又不想把注册逻辑搞得太复杂，导致实作行数太多。有简单的方法吗？还是我就放弃？我就是让它可以依赖，但是只依赖局部的部分的档案。」對 assembly 入口和 migrations 例外答「好，甲」。引文以 `[zcu_lab]` 代替語音輸入的 lab 名稱。
- 2026-11-15，使用者：「但腳本也不應該依賴於我的模擬器環境參數設定，而是自己為每個場景設定的固定環境」，並界定範圍：「研究的话就没差，都可以用。但是如果你要持久的工具的话，麻烦请不要依赖于作者可能会改到的东西。」以上三日的比較見 [Lab 依賴邊界討論](../discussions/2026-11-14-lab-dependency-boundary.md)。
- 2026-11-16，使用者：「檔案大就請不要入庫，用類似快取的方式來減少重複計算，記得考慮跨worktree也能復用的方案」；對範圍與釘住方式答「1a 2a」，對共用快取提案答「R」。見[凍結場景存放](../discussions/2026-11-16-frozen-scene-storage.md)。

### 開發工具的數值交接

**規則。** 量表 caller 擁有選取、reference preflight 與 check 的結束點。獨立 solver 和 cache producer 留 tests，量表只讀 reference，不求解。Check 與量測共用 `tools.family_pareto_reference`，呼叫公開 `reference_key` 與 `load_reference`，不重寫 key。它一次列出所有缺失的名稱與 key，整批成功才開始 policy 或 Engine。

`tools.family_pareto_results` 擁有結果型別、誤差、資格與 dominance；runner、report 和 summary 共用，report 不 import 執行入口。`tools.simulator_measurement` 管理 prepared session 的 mode transition、暖機、history 及 wall／process clock，`tools.paired_benchmark` 才擁有跨 worktree 的 controller、child、ABBA 統計與 telemetry 校準。

**例子。** `family_pareto.select_inputs` 交出工具 `Input` 與原 `FrozenRecord`，metadata hash 用公開 `digest`；預設從共用清單只選已宣告的 Pareto 名稱，讀 payload 前先查重與必需名稱。單列 profile 可顯式選取，但不當 family 去留證據。`family_pareto.measure_family` 提供獨立 reference，將 observations 轉回原量表 schema；計時只包一次 `Engine.evolve`。來源、具名單選、批次 preflight、明確新增列與 exact name set 的契約見 [Freeze numeric scenes](../../tools/README.md#freeze-numeric-scenes) 和[共用清單與 Pareto 選集](../discussions/2026-11-17-pareto-selection-and-scene-append.md)。

**邊界。** 計時 owner 只接準備好的 session callbacks，不 import 場景 producer、solver 或 tests。建構、history reset、暖機、誤差與 telemetry 整理在 timer 外。Sample 建構拒絕負值、NaN 與 infinity；秒數為零原樣記，不補 epsilon 或重取，process CPU 不必小於 wall。

成對比較先確保兩側有相同 committed 工具；controller 走公開來源接縫和 solver-free codec，重用 prepared session，不設轉送層。Warm child 長駐、first／cold 每筆新 process；first 只量一筆，不套慢 family 協定或新增 owner API。Child 沿用外層獨占 runner 的 process group 和鎖。兩側各 clock 的 telemetry overhead 皆在各自 noise IQR 閉區間內才共同開啟。輸出保留原始區塊、reference 證據和完整 density 誤差；額外診斷不混入比值。細節見[共用計時](../../tools/README.md#shared-simulator-measurement)與[成對計時](../../tools/README.md#paired-engine-benchmark)。量測 consumer 不寫 records 副本、不保留舊 freeze helper 或 import producer。測試側 exclusion estimator 及唯一 caller 測試已退休，不留替代 module，不再由該處反向 import tools。

**依據。** 選取、寫入與角色分工見[共用清單與 Pareto 選集討論](../discussions/2026-11-17-pareto-selection-and-scene-append.md)、[成對 Engine 計時](../discussions/2026-11-16-paired-engine-timing.md)、[共用計時](../discussions/2026-11-16-shared-simulator-measurement.md)及 [[0078]]。淘汰的介面與理由見[量表 cache-only 遷移](../discussions/2026-11-16-family-pareto-cache-only.md)和 [Exclusion estimator 退休](../discussions/2026-11-16-exclusion-estimator-retirement.md)。

### Lab 定義與組合根

**規則。** Lab 擁有宣告與模組佈局。唯一入口 `zcu_lab.assembly.lab_definitions()` 回傳 frozen `LabDefinitions`，只取得宣告，不註冊 components、不開 entry、不啟動 GUI 或 MCP。Lab 從已安裝的 `zcu_lab` 與 `zcu_tools` 定位 package roots，並提供 reload 邊界和額外 logging namespaces。九個欄位見 [Lab 組裝與註冊](../../zcu_lab/README.md#lab-組裝與註冊)。Framework 的四個入口各自擁有只含所需欄位的 Protocol：

| 入口 | Protocol owner（省略 `zcu_tools` 前綴） | 欄位 |
| --- | --- | --- |
| Measure GUI | `gui.app.measure.lab.MeasureGuiLab` | `register_components`、`register_experiments`、`templates`、`workflow_attachments`、`catalog_reload_policy` |
| Fluxdep GUI | `gui.app.fluxdep.lab.FluxDepGuiLab` | `register_components`、`fluxdep_export_planner`、`component_flux_context_provider` |
| Dispersive GUI | `gui.app.dispersive.lab.DispersiveGuiLab` | `register_components` |
| Measure MCP | `mcp.measure.lab.MeasureMcpLab` | `register_components`、`recipes` |

**例子。** 三個 GUI 根解析 CLI、取 lab 宣告，用 `functools.partial` 綁 behavior 參數。Launcher 先建 logging，再呼叫 zero-arg factory，讓註冊與建構錯誤進 session log。`logging_namespaces` 直接取 lab tuple，不另設 Protocol。兩個 analysis behavior 先註冊元件，再按 `AnalysisSelection` 開 target；selection 只轉換 CLI，未選 entry 不開 target，只有 component 沒有 entry 就報錯。註冊與開啟錯誤原樣傳遞，不藏在選擇 callback 或保留舊 `project` keyword。API 見[程序 runtime](../../lib/zcu_tools/gui/README.md#process-runtime-runtimepy)。Measure MCP 用 `main(lab, *, gui_module)` 註冊元件及取 recipes；Fluxdep／dispersive MCP 只接 GUI module，直到啟動 GUI subprocess 才由該 GUI behavior 註冊。

**邊界。** Lab 與合成 lab 以結構符合 Protocol；`CatalogReloadPolicy` 由 measure owner 定義，供兩者使用。C16 禁止 framework import 具體 lab dataclass；C13 禁止 MCP 為分享 lab 型別新增 GUI import。也不把九欄搬進共用 framework lab package，否則不需要的入口會間接取得 measure catalog、workflow 或 fluxdep policy。`lab-assembly-only` 允許間接載入，不允許 `apps`、`mcp_servers` 或 `tools` 直接 import 其他 lab 子模組。新增 lab 直接子模組時補 `forbidden_modules`。組合根不重做 lab 組裝，仍決定本次程序選用的 startup 腳本、project root、control port 與 log 路徑，見 [[0064]]。

**依據。** 2026-10-28，使用者：「認同這個建議」。原裁決未附入庫討論，見 Git `cbac6041d`；直接 import 邊界的後續討論見 [Lab 依賴邊界](../discussions/2026-11-14-lab-dependency-boundary.md)。

### 使用者元件與模擬環境的放置

**規則。** `zcu_lab.components` 以 `base.py` 放基礎元件、共用欄位與 resonator wiring 型別，以 `qubit.py` 放共用 qubit 定義與轉換；Fluxonium、transmon、resonator、JPA 和 mux 各有子模組。入口顯式匯出並提供 `register_all()`，不在 import 時註冊。GUI／Notebook 共用的模擬參數與 startup 放 `zcu_lab/simulation_envs/`。每個 startup 與 kickstart 各持同目錄的 `entry-template.yaml`。

**例子。** Resonator 自宣告 `WiringSchema(ch, ro_ch)`；qubit 經 `QubitBaseSchema` 用 `QubitWiringSchema(ch, lo_flux_ch)`；JPA 與 current source 沒有通道讀取者就不宣告 wiring。Mux 用自己 schema 的 `gen_ch` 和有序 `ro_ch`，不從單 tone resonator 推導。有值時檢查 ADC 與 resonators 等長，unknown wiring 的契約見 [Components README](../../zcu_lab/components/README.md#wiring)。GUI 執行 caller 選定的可信 startup，不探索目錄；Notebook 可直接呼叫 `experiment.ftf.ftf(...)`，FTF 不設離線建庫入口。`zcu_lab/kickstart/<env>/` 只建 entry、元件參數及 alias，不連 SoC 或 device；改一份 template 只影響使用該腳本的新 entry。

**邊界。** `BuiltinComponent` 只給 `ext`、`module` 與 unsupported-field migration。C14、C16 禁止框架反向 import 使用者定義。Workflow 自組 program 並呼叫 runner，可沿用實驗 core 的 Cfg、Result 與 `AXES_SPEC` 存檔，不借用其他實驗的取數和分析。專屬 helper 留 workflow 目錄，共用 helper 放 `zcu_lab/workflows/common/`，不把客製需求放進 core 或 lib。`@recipe` 與 `@workflow` 同形；recipe 由 `zcu_lab/recipes` 顯式登記。Startup、recipe 和 workflow 的作者慣例見 [lab policy](../../zcu_lab/POLICY.md)。

**依據。**
- 2026-10-16，使用者：「比如components我希望除了基礎原件，其他的都是獨立的子模塊檔。」並確認「可以移到zcu_lab/simulation_envs/中嗎？2同意」。
- 2026-10-21，使用者：「3我希望startup腳本間獨立，而不是過早共用template」，並定目錄「既然都叫kickstart了，那我認為可以將envs/FTF改成kickstart/FTF」。同日較早的 workflow helper 放置方式已被下述裁決取代，原話見 Git `9b4b9812c`。
- 2026-10-23，使用者：「mux group的屬性沒有寫好，simulation/FTF 的startup忘了寫mux group的gen_ch，而ro_ch沒有意義，因為有多個ro_ch」，並問「為什麼BuiltinComponent有那麼多無用的欄位，他不該是……儘量空的嗎？」
- 2026-11-01，使用者：「所以我才不想要一個實驗使用另外一個實驗，twotone freq與workflow用的helper應該要獨立實現需要的流程」，並說「3a可以，b是根據一個設計理念，耦合的邏輯就該放在同一個位置，這樣改邏輯的修改範圍才局部」。原裁決未附入庫討論，見 Git `a56424f46`、`695553673`、`ad9e6f811` 與 `e97bead1c`。

### Library 分組與 module slot

**規則。** Library 的設計是元件 → 分組 → Pydantic model。Module slot 指向下游實驗要使用的模組，不為每一種校準方案另存 slot。單音 readout 與 readout slot 歸 resonator，不歸 qubit。Qubit 經自己的 `resonator` 欄位與 session 的 `resonator: $qubit.resonator` alias 取得讀出。

**例子。** `R1.module.readout` 可指向 `R1.readout.readout_rf`。`twotone/ro_optimize/*` 把最佳化 module 寫到 `$resonator.readout.readout_dpm`，並提議更新 `$resonator.module.readout`。Apply module 和 slot 後，使用該 resonator 的實驗都沿此 slot 取最佳化讀出。`$qubit.ext.best_ro_freq`、`best_ro_gain`、`best_ro_length` 的位置與用途不變。

**邊界。** 現行格式只接受四個分組，有第五種需求才改容器與 key 驗證。`module` 是可擴充的 `dict[str, str]`，增 slot 不改 component schema。單 qubit 表單不讀 qubit 的舊 readout slot；resonator 無 slot 時用 readout 範本，不遷移既有 point。如何累積參數、哪些實驗提參數、哪些收尾實驗更新 module，由作者決定。Framework 不提供通用齊全檢查或累積機制，欄位、reset 名稱、pair 方向留在 lab 作者文件。

**依據。** Library 分組原裁決日期未記錄。使用者原話「我無法知道元件下會需要什麼欄位，我認為元件->xxx->pydantic model比較適合，不規定只有四種」及「我選D，但暫時沒有超過這四種的需求」，是設計方向，不是立即改格式。2026-10-16，使用者在 map 結案時以「都採用建議」確認沿用方向；同日以「都採用建議」確認 module slot 的歸檔範圍。2026-10-25，使用者：「目前onetone/freq的readout module writeback是寫到qubit裡，但我認為應該是寫入到resonator componenet裡。而qubit要使用的readout應該是去綁定的resonator中找才對」。對方案及遷移答「選甲，不遷移，應該說，目前module欄位下應該沒有限制型態吧？應該可以擴充吧？」同日以「Readout採用建議」確認最佳化讀出也寫進 resonator 並換掉 slot 指標。原裁決未附入庫討論，見 Git `854f4198b`、`e150a6c6d`。

### Mux group 是實驗室的 component 定義

**規則。** Lab 在既有 ComponentRegistry 註冊 `MuxGroupSchema` 為 `group/mux`，不加 framework registry、RPC 或保存格式。Group 擁有 readout module 及有序 resonators。完整 readout 的 `tones[i]` 綁定 `resonators[i]`，重排不改 module owner。Lab 將 cfg 的 participants 原始索引按輸入順序交給 program／raw 的 `active_tones`。Program 不知道 group 或 resonators。RunRecord 存完整 readout 與 selection，資料軸不重編號。Cfg 不存 group snapshot 或 writeback 目的地。

**例子。** Scalar alias `mux=MG12` 讓 adapter 從固定 DocumentSetCapture 讀 `$mux.resonators` 和 `$mux.module.readout`。一般 mux 與 FTF 有 slot 優先用 library 的完整 tones 與 scalar 設定；一般 mux 無 slot 用 blank 加 group tones，FTF 八張表單用 blank，rb 仍 Invalid。Mux onetone 在副本套 group tones 和 ringup 初值，缺 qubits 也可手填 readout。Mixer 初值取 group 的有限 LO MHz，不從 tone 頻率或 κ 猜。Group 與 resonator 的 `time_of_fly` 是各自路徑的頂層有限非負 µs，不讀 ext、不自動遷移。Group TOF 或任一 κ 缺值時，無 slot 的 mux onetone 將 trigger 和 pulse length 留空必填；不用單 resonator 路徑的 TOF。初值公式與欄位見 [mux readout 與表單初值](../../zcu_lab/v2/README.md#mux-readout-與表單初值)。

**邊界。** Model 驗證 1 至 8 個 resonator 名稱、選填 qubits 等長且不重複；`resonators` 取代 `members`，無相容讀取，也不反推 qubits。選填 `flux_dev` 只記預設 global coil，不綁 qubits 或保存各 qubit 校準。Global fluxdep 分別提出 `<qubit>.flux.period.<device>` 和預設勾選的 `<qubit>.flux.offset`，append 原始擬合 record，沿 [[0082]] 的 flux 定義及 [[0063]] 的單文件 mapping-leaf 合併與交易。Group 也容許未知 wiring、`mixer_freq` 或 `time_of_fly`，文件允許 `wiring: {}`，缺 ADC wiring 時各 tone 表單 `ro_ch` 留空必填。Schema 與 I/O 分界見 [lab 組裝](../../zcu_lab/README.md#lab-組裝與註冊)及 [wiring](../../zcu_lab/components/README.md#wiring)。

提供 qubits 時，lab Form 在 Open／Reset 用固定 capture 檢查兩邊元件存在及 kind，逐項核對 `qubits[i].resonator == resonators[i]`，缺失或 null 也拒絕。錯誤指出欄位、index 與實際值，不排序、猜配對、修正或另拒重複 resonator。通用 alias owner 解析 `$participants` 的有序名稱，lab 負責從固定 capture 建立 participants。轉換時機、原始索引及 tone 初值見 [ADR-0079 的固定具名元素與參與清單](0079-form-field-transitions.md#固定具名元素與參與清單)及 [lab 的 Mux 參與名單解析](../../zcu_lab/v2/support/measure/README.md)。Program cfg 的 `active_tones` 驗證沿宿主 result／Run 診斷拒絕 Run，不保證欄位變紅；preview／describe 不 lowering，FormEditor 不投影外部錯誤，見 [[0065]]。

Adapter 的 writeback 只從 Run／Load capture 解析 `$mux.resonators` 和成功 active tones。不能猜位置或讀目前 alias。缺 capture 只提 module 並顯示原因；capture 可解析時也提 `$mux.module.readout` 指向新 root，完整 tone 數與 resonators 不符時仍保留 module 與 slot 提案，但省略逐 resonator scalar 並說明。新 qubit 實驗須核對全 tone 數才投影 active tones，成功 tone i 寫 `qubits[i]`；onetone 仍寫 `resonators[i]`。固定 module target 是 `$mux.readout.readout_rf`，與當前 slot 無關。Framework resources、session、program 不解讀這些關係，具體 cfg 與 recipe 由 lab 接線。依賴沿第 4 層到下層、C15 與 C16 不變。

**依據。**
- 2026-10-16，對專屬 group component 答「我選B」，對 scalar alias 答「B」。使用者提議「一個MG component類，內部屬性綁定到R1, R2，gui中預設指定MG，寫回目標也是從MG的綁定取」，並補充「為了能只讀特定幾個R，可以考慮MG外cfg另外提供mask之類的」；對 active_tones 答「可以」，對報錯答「報錯我選B」，對寫回答「應該參考其他使用alias是怎麼寫回的」。
- 2026-10-18，使用者：「或是改寫mux_group這個原件的欄位吧，qubits分resonators」。必填 resonators、選填 qubits 並逐項檢查、不留舊欄相容讀取，答「12都同意」。
- 2026-10-20，使用者：「不只twotone，應該說除了onetone與readout optimize/mist等本身就在處理readout相關，其他實驗的readout預設值請優先使用ref。」
- 2026-10-23，使用者：「mixer frequency屬性請加入到mux group中，並用於生成預設值。」「讓mux group有自己的time_of_fly屬性，也用於生成預設值。」「將time_of_fly轉正吧。不在放ext裡了。」同日指出「但kickstart應該是生成欄位讓用戶填，必填的話無法做到」。原裁決未附入庫討論，見 Git `b529634c7`、`31aa46de3`、`c3d979805`、`244ec7aed` 與 `753c0c1d3`。

### 能力模組的兩類

**規則。** 依賴指向穩定的一方。被許多模組依賴的模組應少變，常加功能的模組不供其他模組依賴。

| 類別 | 模組 | 規則 |
|---|---|---|
| 穩定工具與值型別 | `core`、`cfg_model`、`utils`、`analysis`、`simulate`、`plotting`、`datafile`、`progress_bar` | 任何模組可使用；工具之間可依賴但不可循環 |
| 有狀態的能力模組 | `program`、`resources`、`device`、`qick_remote` | 彼此不 import，以 `cfg_model` 的型別與 Protocol 溝通 |

**例子。** 只用 fitting 時不因 `analysis` 初始化載入 fluxdep 的互動 line picker。`utils` 只在 `format_obj` 轉換時載 QICK；`datafile` facade 按公開名稱載入 owner，storage 用 path helpers 不會載 HDF5。公開名稱、計算、保存契約不受影響。

**邊界。** 純工具接收與回傳數值，不自行讀容器、硬體或其他外部狀態；檔案操作限其既有責任。`simulate` 甚至不依賴同列的穩定工具；見[封閉子模塊](#模擬器是封閉的子模塊模型由使用方聲明)。C8、C9、C12 的方向不變。

**依據。** 分類沿用本篇的 package 定位與 `.importlinter`，專用依賴延後載入見 Git `e2570b14a`。原分層裁決未附入庫討論。

### 共享核心 core 與 cfg_model

**規則。** `core` 只依賴標準庫、外部函式庫與 `utils`，擁有共用的無狀態表示轉換與 waveform 格式。`cfg_model` 擁有 `ConfigBase`、module／waveform cfg schema 及有狀態模組共同需要的契約；型別至少有兩個有狀態模組使用才進 `cfg_model`。Resources 解讀 formula、管理檔案並用核心 schema 做 DocumentStore model；program 將 cfg 轉為執行型別並依 `type`／`style` 選 builder，模擬器用編譯後的 envelope。

**例子。** `normalize_numpy_scalar` 為 AnalysisDocument 與 AcceptedWrite 轉換 numpy scalar，不遞迴。Waveform samples、hash、NPZ codec 及單一 `WaveformFormatError` 由 [core](../../lib/zcu_tools/core/README.md) 擁有，recipe 只驗 JSON 語法。`cfg_model` 的 schema 欄位、驗證、`set_param` 及 `ModuleResolver`／`WaveformAssetInfo` 見 [cfg_model README](../../lib/zcu_tools/cfg_model/README.md)。Resources 實作契約，驗證 context 注入固定 library 和資產能力。`ModuleCfgFactory.from_core` 以 `assets.path_for` 將持久 arb 轉為含 data key 和絕對 path 的 native cfg，不留 provider、不查 global DB。CZ 優化候選直接用 `InMemoryArbWaveformCfg` 的唯讀樣本，不存入資產庫；採用的最佳樣本仍用既有 NPZ arb 格式保存。

**邊界。** Finite／strict JSON 驗證留在 datafile、resources 各自邊界，方向沿 C8、C9、C12，轉型只需 numpy，符合 C19。`cfg_model` 只依賴外部函式庫、utils、`core.waveform_format`、`core.complex_codec`，不做 I/O；classifier 複數中心共用既有 tag codec。`cfg_model` 和頂層 `format_version.py` 仍在原處，搬遷排在 waveform 接線後，尚無排程。持久 library 不接受 native `arb_memory`，它沒有 key 或 path，不能序列化。兩種 native arb 共用 envelope 取樣，檔案 arb 以 channel＋path、記憶體 arb 以 channel＋樣本雜湊共用。

核心 schema 寫進檔案，例如工作點的 `library.yaml`。Library 的 format／format_version 只管 header 和容器，不因 Spec 欄位變更自動升版。Library 維持 2.0，不遷移或加相容讀取；entry 驗證現行 schema，不以 forward minor 放寬 extra。Library 逐條驗證，個別 Spec 過時不影響合法條目，略過無效 Spec、查原因及讀取不改檔見 [[0063]]。

**依據。** 2026-10-12，使用者：「採 1：core.numeric_scalar.normalize_numpy_scalar 放 core（同 complex_codec 的無狀態表示轉換），finite/strict JSON 驗證留在兩個 boundary，core README 補一行。」2026-10-15，使用者：「只是改Module而已真的需要升Library版本嗎？就像加一個Component原件定義，不應該讓舊版本不能讀對吧」「不要一次驗證，而是best-effort」「1可以，2A，3保留2.0」。原裁決未附入庫討論，見 Git `c0158f785`、`fbcb48e91`。2026-11-19，使用者：「我不希望的一个情况是，随机演算法中途的波形被存到永久的资产库里」，對 program 直接收樣本答「可以」。見[記憶體 arb 討論](../discussions/2026-11-20-in-memory-arb-waveforms.md)。

### 分析 app 的 entry/component owner

**規則。** Fluxdep／dispersive 共用 `gui.app.analysis_project` 的 entry／target 選擇、EntryDialog、CLI 與 RPC 投影。兩個 app 都只接受 fluxonium 類目標。`analysis.fluxdep` 擁有 Container 到 native 光譜的數值投影；`gui.app.analysis_project` 擁有 native source 讀取、variable／axis 預選與 dialog mechanics。Predictor 與 dispersive 格式由 resources 擁有。

**例子。** Roots 經 lab 描述把 kind registrar 交給 analysis behavior。Behavior 先註冊，再依 `project_root` 下的 `result/` 與 `Database/` 開既有 ResultEntry；省略 component 時用 global `qubit` alias，缺失或錯誤則由 dialog 要求明選，不自選第一筆。Fluxdep 匯出 predictor 並更新 global 已宣告欄位，以 `record_analysis` 留 ledger；dispersive 讀 entry-local predictor，匯出獨立 artifact。介面見 [analysis project README](../../lib/zcu_tools/gui/app/analysis_project/README.md)。Caller 明示 variable、device_axis，`project_native_spectrum` 驗 2-D 複數資料和另一軸的頻率單位並轉成 MHz，兩 app 不各猜布局。

**邊界。** 舊 chip／qubit 目錄由 migrations 處理，app 不相容讀取、不呼叫 Database Save。Notebook 的 `params.json` 仍歸 `resources.qubit_params`，不是 app 的 fallback。Native source dialog 只接 caller 的 device choices，不取 component 或 predictor，不含 fluxdep 的 Type、Inherit 或 overlay。Framework 不 import lab。共用 GUI、session、remote、measure、workflows 不依賴分析 owner，由 `shared-no-analysis-project` 檢查。`nearest_existing` 歸 `gui.widgets.file_dialog`，不由分析 owner re-export。

**依據。** 2026-10-20，原裁決記為「新增 gui/app/analysis_project/ 共用過渡 owner」。2026-10-20，使用者：「1B可以，2要，3讓migrations裡支援遷移，工具本身不兼容」。2026-10-23，使用者：「fluxdep／dispersive 維持 fluxonium 專用」。Native 光譜的分工由 advisor 於 2026-10-27 依同日授權「C也是，可以直接派」代裁，未見使用者覆核這項分工的紀錄。原裁決未附入庫討論，見 Git `e9b408317`、`be108f5d2`、`257e8cc3a`。

### Device snapshot 標量的解讀 owner

**規則。** `device.snapshot_scalar_values` 是 acquisition snapshot 到 `DeviceScalar(value, unit)` 的唯一解讀入口。`analysis.flux` 擁有換算時的單位比對，apps 不解讀 driver settings 或 cfg。

**例子。** 儀器名稱缺席就不回該鍵；名稱在但 value 缺漏、為 bool 或非有限，報錯並指名。缺 unit 視為 native。詳細契約見 [Device README](../../lib/zcu_tools/device/README.md#實驗控制與-acquisition-snapshot)及 docstring。

**邊界。** 被掃儀器的值來自掃描軸，不重用 snapshot 值。只比對實際參與換算的其他儀器，單位須完全相同，native 只等於 native，不換算。整份 snapshot 未記錄不等於已記錄但個別儀器缺席；個別缺席依 [[0082]] 貢獻 0。

**依據。** Advisor 於 2026-10-27 依同日使用者授權「C也是，可以直接派」代裁，未見使用者覆核這項分工的紀錄。原裁決未附入庫討論，見 Git `257e8cc3a`。

### 模組之間的合作

**規則。** 能力屬於別的模組而該模組不提供時，擴充 owner 或停下來問。不在 caller 重做。兩個有狀態模組要合作，由需求方定義最小 Protocol，組合根注入實作，或把組合放到上一層。

**例子。** Caller 如果驗證 callee 的輸出格式、組 callee 的內部版面或修補其回傳，先調整責任，不複製實作。

**邊界。** 只有實作由外部決定才注入。框架常用的支援直接放內部子模組；使用者腳本為保持獨立可以自行集成算法。

**依據。** 2026-10-20，使用者：「C太多注入也不是很好維護，有些可以是框架提供支援的幾種功能，已內部內聚子模塊實現，而不是用注入的方式。script有些是為了獨立性而集成算法。」原裁決未附入庫討論，見 Git `a501b9352`。

### 可執行投影

**規則。** [`.importlinter`](../../.importlinter) 投影本篇的依賴方向，編號 contract 如下：

| contract | 類型 | 內容 |
|---|---|---|
| C8 | `layers` | 前端框架 > `experiment` > 能力模組 |
| C9 | `acyclic_siblings` | `zcu_tools` 頂層 package 無循環 |
| C10 | `independence` | `program`、`resources`、`device`、`qick_remote` 彼此獨立 |
| C11 | `forbidden` | `cfg_model` 不依賴有狀態模組、實驗及前端；可用 `utils`、`core.waveform_format` 與 `core.complex_codec` |
| C12 | `forbidden` | 穩定工具含 core，不依賴有狀態模組 |
| C13 | `forbidden` | `mcp` 只使用 `gui.remote` 與各 app 的 wire spec |
| C15 | `forbidden` | 使用者實驗 core 不依賴 GUI framework |
| C16 | `forbidden` | `zcu_tools` 不 import `zcu_lab` |
| C19 | `forbidden` | `core` 在 repo 內只依賴 `utils` |
| C21 | `forbidden` | `zcu_tools` 不依賴使用者入口 `scripts` |

**例子。** `composition-roots-above-framework-and-lab` 檢查 `apps : mcp_servers` > `zcu_lab` > `zcu_tools`，兩根不互相 import。三個 GUI 根與 measure MCP 根各注入 lab 描述；兩個 analysis MCP 只注入 GUI module 名稱。`lab-assembly-only` 檢查根與 tools 只直接 import lab 的 assembly。`experiment-directories-independent` 檢查實驗和家族邊界，不新增 `ignore_imports`。`simulate-closed` 像 [[0080]] 的 `datafile-standalone`，明列其他頂層 package 與 `format_version`，不用 `ignore_imports`。

**邊界。** Import-linter 的 roots 是 `zcu_tools`、`zcu_lab`、`migrations`、`apps`、`mcp_servers`、`scripts`、`tools`。`migrations` 與 `scripts` 不在 `lab-assembly-only` 的 source_modules。離線程式與對照放 `migrations/`，lab 和 lib 不 import，遷移完成後可整個刪除。Tests 的 lab 獨立性用 review 檢查 import，必要時用 grep，不新增 tests contract。債務的 `ignore_imports` 按確切 module 對列出，不用萬用字元，註解寫明「債務，由誰移除」，只減不增。

**依據。** 2026-10-14，使用者：「我自己會預期，我遷移完後刪除migrations就可以了」。原裁決未附入庫討論，見 Git `00f18a5b6`。現行依賴方向見上述設定；assembly 和封閉子模塊的討論分別見[使用者空間](#使用者空間的依賴原則)與[模擬器邊界](#模擬器是封閉的子模塊模型由使用方聲明)。

## 債務

| 債務 | 移除方式 | 記帳位置 |
|---|---|---|
| `program → resources` | `program` 改依賴 `cfg_model` 的 Protocol | C10 `ignore_imports`：`program.v2.modules*` → `resources.context` |
| `notebook/analysis` | 通用分析移到 `analysis`；實驗專屬 helpers 已由 `zcu_lab` 附件擁有 | 尚未搬移 |
| `utils` 的 `shot_classification`、`tomography` | 移到 `analysis` | 尚未搬移 |
| `analysis/fluxdep/io.py` 直接用 h5py | 檔案讀寫移到 `datafile` | 尚未搬移 |

## 後果

- 增減功能時，工具之間的依賴很少變，有狀態模組之間不新增直接依賴。兩個有狀態模組合作時，以 Protocol 接線或上移組合。
- 新的跨 package 依賴會被 contract 擋下或由 review 檢查，不會因留在程式中就成為先例。
