---
status: draft
---

# 模擬環境、FTF preset 與參數檔

**狀態：** 已核准方向的設計草稿。明確環境、FTF 分段引擎及重算式單 round 重播已交付，現行入口見各 owner README。F1／F2 校準已跑通，conditional Ramsey 取得一點，依「能跑起來」標準接受。這些歷史證據屬舊 backend，並非目前版本的精度或成本保證。本文保留使用者裁決原話，不以早期候選敘述替代現行 owner 契約。

**關聯：** [[0064]] 啟動組裝、[[0068]] remote transport、[[0069]] 實驗核心與前端、[[0071]] coherence 單位、[模組定位](../0072-module-boundaries.md)。模擬入口與責任以本文及各 owner README 為準，舊 TLS 遷移已完成。早期選項比較保留於[討論快照](../../discussions/2026-10-13-simulation-environment-ftf-preset.md)。

## 目的

讓使用者經既有 GUI／MCP 量測、Run、分析與保存流程，練習 FTF 的工作點搜尋、conditional phase 與 leakage 校準。FTF 是 fluxonium–transmon–fluxonium，transmon 是 coupler。第一版不以精確預測真實晶片 gate fidelity 為目標。

2026-10-13，使用者指定優先順序：「提醒你，我們要討論的是在既有計算能力與物理計算模型與實作複雜度中，儘可能納入各種現實會有的現象。FTF會著重在two qubit gate與coupler上，單qubit的可以模擬的比較粗糙」。因此先保留 gate／coupler 的主要物理與可量測 I/Q，再權衡計算與實作成本，不以完整 Ramsey／echo 包絡或所有微觀 bath 為前提。

2026-10-14，使用者提出 white_noise、single_fluxonium、single_SQUID_transmon、FTF 的預計 preset 清單。對名稱確認及是否四個都在第一版交付，回答：「1是 2先FTF，架構設計保留擴充」。2026-10-15，使用者調整範圍：「也加入white_noise，請繼續」。因此第一版交付 FTF 與 white_noise，single_fluxonium、single_SQUID_transmon 仍是後續方向，不納入本版驗收。White_noise 不含量子演化，仍使用共同環境與 acquisition 契約；雜訊分布與幅度定義見下節。

擴充沿用共同 build_environment 回傳、request 與 signal 契約，替換模型與組裝接線。Program frontend／adapter 與 GUI 不寫死 FTF 的元件數、讀出配置或 YAML 欄位。FTF 自己擁有其模型限制；不為後續 preset 預建空 class、registry、plugin discovery 或通用 DSL。

完整 acquire 平均 1 秒／sweep point 是效能目標，不是已驗證保證。首用成本計入模型初始化、控制解析、prepare、演化、讀出、抽樣與 packing，warm 成本另列。Reps／rounds 不作分母。不能引用舊 M=8、toy gate 或 frozen-state readout 的速度宣稱達標。

## 決策

### Python 是環境入口，preset 的 YAML 是參數

環境以 `--simulate <path.py>` 在啟動時選定，只呼叫一次 `build_environment()`。Python 組合物理模型、模擬資源、接線與 backend；不同拓樸不必填入同一固定 engine。FTF preset 自行讀 YAML，一般使用者複製範例並調整數值即可換樣品。

2026-10-14，使用者澄清：「但我預期模擬參數yaml中能夠設計ch，也就是說，build_enviroment回傳應該包含mock soc」。FTF builder 依 YAML 的 channel 設定組裝可執行的模擬 SoC、匹配 soccfg 與模型接線。GUI／Notebook 直接使用這一對物件，不另外呼叫預設設定 helper 建立第二份 soccfg。具體 channel schema 由實作完成。

2026-10-16，對固定虛擬 QICK capability profile、只開放 channel 數量與接線的建議，使用者確認：「可以，請定稿spec與拆tickets，但不要實作，我會交給獨立agent實作」。第一版使用一套明確支援的能力規格，不接受任意 clock、DDS、channel type、memory limits 或整份 QICK dict 覆寫。實際使用未綁定的 channel 時明確報錯。這保留環境接線能力，不先建立通用虛擬硬體設定框架；具體 profile 數值由實作明訂，不把研究候選值當預設。

同日，使用者提出：「可以有一個make_mock_soccfg(設定參數)，讓build_enviroment解析YAML並傳入。然後board是什麼？為什麼不直接soc？」對移除 board 包裝、環境直接提供 soc／soccfg 的方案回答：「好，採用，繼續」。Builder 讀取 YAML，將具名設定交給 program 的 make_mock_soccfg，再用同一份 soccfg 建立模擬 SoC。此設定入口不讀 YAML 或解釋 FTF 物理，不因保留名稱而保留舊無參數介面的相容層。

YAML 不是通用 environment DSL。Launcher／runtime 不辨識 FTF 欄位，不建立 registry、設定繼承、plugin 自動探索或 hot reload。環境檔是可信 Python，不提供 sandbox。

2026-10-14，使用者比較傳入 controller 與回傳環境兩種方案，裁決：「考慮到要與gui解耦，我選第二種方案。」build_environment() 不接收 GUI controller，回傳具名結構中的已組裝 Python 物件。GUI／Notebook 共用組裝入口，再由各自資源 owner 接管裝置與模擬板。宿主不解析 FTF 參數或依描述重新建立 backend，不繞過 owner 修改 registry／State。

同日，對「直接回傳已填好的 DeviceManager，並以 build_environment() 成功回傳作為清理責任交接點」的建議，使用者回答：「好，沒意見」。Bundle 直接包含已綁定的 soc、匹配的 soccfg、已註冊具名裝置的 devices: DeviceManager，以及不含模型真值的公開 description，不再增加 board／SimulatedBoard 容器。GUI 注入同一 manager，由 DeviceService 發布裝置狀態；Notebook 直接使用它。不搬移 driver 或重新註冊，避免兩個 manager 同時負責關閉同一裝置。

Builder 回傳前失敗，由 builder 清理已建立資源；成功回傳後，由宿主負責安排清理，不等 GUI State 發布成功才交接。安裝失敗時，宿主透過同一 manager 清理本次資源，不開放 Run／remote readiness，也不回退正常模式或白雜訊。部分 close 失敗保留失敗裝置與錯誤，不自動 retry，不宣稱清理成功；原始啟動錯誤與清理錯誤都保留。

啟動失敗的清理不能只依賴成功顯示視窗後的關閉事件。正常關閉仍遵守既有 operation／Run gate，不因持有 manager 而繞過使用資源的工作。`SimulationEnvironment` 回傳同一 manager，`DeviceService` 先發布 live devices 再還原偏好。同名 live 不被替換，其餘 remembered devices 保持離線。這不新增通用交易框架。

2026-10-13，使用者提出：「我覺得可以有FTF的enviroment或preset，內部有個yaml檔來放一些超參數，這樣方便用戶調」。對 Python 組裝、YAML 調參的分工確認：「好，繼續，同時你可以將我們的討論記錄到docs/discusssion中了」。更早的原則原話為「我第一版想做簡單一點的，不要做出什麼複雜的DSL架構，可以考慮宣告本身是python腳本」，原始日期未保留，於2026-10-13彙整。

### 啟動固定模式與可信程式

正常／simulate 模式不在程序中途切換。載入、驗證或安裝失敗就中止，不回退正常模式、白雜訊或半成品環境。MCP launch 傳遞同一環境路徑；attach 不改模式或重新組裝。正常模式不代表自動連硬體。

2026-10-14，使用者對直接移除 GUI Use Simulate Env 與 MCP simulation.initialize 回答「同意」。不保留相容入口或同名隱式重啟；切換環境、修改 YAML 後須明確重啟。裝置目前 bias 的合法修改是量測輸入，不是重載環境。Memento、Setup 或 RPC 不改啟動模式。

同日，使用者裁決：「先不必拒絕真實資源，我自己會採用主機隔離的安全方案。」因此不新增依 simulate 模式拒絕真實資源的連線／操作政策。既有 Run lease、控制身份與 mutation gate 照常生效；合法實驗內部控制不因這次設計被禁止。

### 模型參數、裝置設定與量測輸入分開

| 資料 | 責任與生命週期 |
| --- | --- |
| 模型真值與校準係數 | 描述電路、耦合、耗散、線路與讀出，啟動時載入並驗證。 |
| 數值設定 | 基底、截斷、步長與容差，和物理參數分開；不承諾任意值都足夠。 |
| 裝置初值／目前值 | 初值只在啟動套一次；每 round 使用當下裝置值。 |
| Program 量測要求 | Pulse、gain、frequency、phase、timing、sweep、readout 與抽樣要求；YAML 不提供隱藏 pulse 來覆寫它。 |
| 校準估計 | 從 I/Q 推得，可用於後續量測，不直接改模型真值。 |

Loader 驗證必要／未知欄位、units、維度與數值合法性，失敗中止；solver 只收具名數值，不讀 YAML 或資料容器。`resources.ftf.FTFSettings` 定義具名 schema 與預設值，`parse_ftf` 提供 field-path 診斷。範例參數不代表任意模型的精度保證。

2026-10-14，使用者提出：「可以考慮讓FakeDevice在建立時設定使用的參數，預設是native這樣」。對建立時指定且建立後固定的提案，回答「認同，再來要做什麼？」。FakeDevice 接受 value_unit=native／V／A，預設 native；driver 保存單位，Info 如實回報。Value 與 rampstep 使用同一單位，不自動換算。Setup 不切換單位，傳入不符單位時在寫入前拒絕；要換單位就重建裝置。

FTF builder 明確選 V／A，flux helper 比對實際 driver 單位，不以預設 native Info 代替。GUI／recipe 沿用裝置資訊，不新增 FTF 專用 DC driver 或物理分支。這不保留舊 TLS，也不決定白雜訊 mock 的去向。

2026-10-14，使用者對 DC／pulse 分開校準回答「同意」。控制映射為 `φ(t)=φ_offset+M_DC u_DC+φ_pulse(t)`，φ 是 F1、coupler、F2 的 Φ/Φ0。u_DC 來自該 round 的電壓／電流，M_DC 每欄明示 Φ0/V 或 Φ0/A，可有串擾。Pulse 另用 Φ0/gain 校準，先套來源線 delay／一階低通，再映射到 loops。實際 flux 進 Hamiltonian，不直接改寫頻率答案。

同日，使用者對 microwave、flux pulse、readout 的物理 gain 定義回答「1認同 2同意 3可以」：

- Microwave：`H_d/h=Σ_i,s Re[c_is g_s w_s(t) exp(i(2π f_s t+θ_s))] n_i`。c 為 GHz/gain，f 用 GHz、t 用 ns、θ 用 radians，保留完整 charge operator 與 signed complex envelope。不得逐 waveform 重設峰值或以 n01／π-gain 隱藏校準；採 RWA 時由 backend 處理振幅的1/2。
- Readout：`b_in=η g w exp(iθ)`，η 為 sqrt(photons/ns)/gain，program 另提供 carrier。模平方是入射 photon flux，腔內 photon number 是響應結果。Receiver gain／phase／offset／noise 獨立，接收增益不改物理驅動。

### 每 round 捕捉輸入，不承接前次量子態

每次 acquire 固定一份 AcquireRequest，包含控制 timeline、量化座標、readout routes/windows、layout／mode 與 acquire 抽樣身份。每 round 真正執行前捕捉 EnvironmentSnapshot，包含裝置值與所用模型／校準設定版本；兩者加 round 身份組成 RoundRequest。這些具名型別由 `core.simulation` 提供，組合採具名欄位，不使用 dict 覆寫或 live device 引用。

2026-10-14，使用者提出：「我預期在acquire時會記一筆acquire request，而每個round又記一筆環境的參數，這兩筆合成一個模擬的request，這樣合理嗎？」對此拆分確認「好，繼續」。原因是「因為可以預見一種使用場景，每round間隔改一次flux device值」。下一 round 必須看到新值；不逐 ADC getter 重讀裝置，也不在 prepare_round 過早固定環境。

同一 acquire 的 program／layout 固定，已排定的 sweep／register／table 控制仍執行。不熱載入 YAML、不自動重編譯被改動的 cfg、不放寬外部 mutation gate。不同 flux rounds 的最終平均是混合條件平均，不自動形成 flux sweep，必須保留各 round 輸入以追溯。

模型不承接前一 shot、round 或 acquire 的量子態。每 shot 從當下 DC、含耦合 H0 的共同溫度 Gibbs 態開始；T=0 使用基態。Reps 間 relax_delay 不造成殘留激發或 repetition-induced steady state，shot 內 idle 正常演化。主機等待不推進物理時間。

2026-10-13，使用者接受限制：「2理論上希望能支援reps這層的relax_delay效應，但可能難度太大了，每個shot獨立是可以的。」對熱初態選擇「B，需要考慮Thermal population」，並對初態／耗散使用共同溫度選擇「B」。這不是不支援熱人口，也不是證明跨 shot 演化不可行。

### Acquire 接線與末端讀出的裁決

2026-10-14，使用者要求：「認同，我認為可以考慮建立一個攔截acuqire的ProgramMixin，讓既有program繼承。」並指定「關於Program集成，我希望模擬的認知集中在Mixin裡」。

SimulationAcquireMixin 由 program owner 提供，只在 MyProgramV2 共同基底接入一次。它準備 frontend／adapter 與資料來源，沿 super 委派既有 acquire，不讀 YAML、DeviceManager 或建立 FTF Hamiltonian。一般 MyProgramV2.acquire、具體 program 與 modules.run 不另留模擬分支；不建立 FTF 專用 mixin 或平行實驗類別。

使用者其後縮減範圍：「我目前沒有step_round的需求，有沒有簡單的方案」，並確認同步方案：「好，那用這個方案」。第一版僅支援同步模擬 acquire／acquire_decimated：

- 有 binding 時，在建立 request、消耗 acquire 抽樣序號或安裝本次資料來源前拒絕 step_rounds=True；沒有 binding 時原樣委派，正常硬體分步能力不變。
- 保留 mixin 前置於 ImproveAcquireMixin 的安排，只包住同步呼叫。取得本次資源後以 try/finally 清理資料來源、暫存及重入標記，保留已完成結果、environment binding 與確定性 cache。公開 acquisition 測試涵蓋此 MRO 與型別接縫。
- 同 program／模擬 SoC 只在呼叫期間防重入。拒絕在覆寫原工作前完成，不能進入會清掉原工作的新呼叫清理範圍。
- 不新增 AcquireLifecycleMixin、abort 入口、跨方法分步使用權或整組 prepare／finish lifecycle wrapper。同步 callback 例外會回到外層 acquire。

既有 Run／QICK round loop、hooks、tracker、分類、normalization 與 averaging 保留。EarlyStopMixin／cancel_flag 繼續擁有停止及部分結果語意，不承諾立即中斷同步 solver 或有界停止延遲；實測過久再評估合作式取消。

第一版只支援一個共同末端讀出時段，FTF 使用一個 readout gen_ch 與一個 ro_ch。Mid-circuit measurement、feedback 與需更新量子態的分離讀出時段明確拒絕，不因讀出模型包含 relaxation 而擴大此範圍。

### Module frontend 與小介面依賴

2026-10-14，使用者選擇「我也認為保持Module語義的解析比較可行」，並對 execution context／分段求解回答「可以」。Frontend 解釋 Module 語意，持有局部 register、sweep／loop、table 與時間排程 context，不建立通用 tProc ISA emulator 或全域 register table。未知操作與未定義 register 明確失敗。

同日，使用者指定放置：「採用建議，然後考慮到這個模擬機制高度依賴program的Module，所以我認為其位置應該可以放在program/v2中的子模塊，但其他program模塊不可以依賴他，這樣方便維護的同時保持解耦。」隨後澄清：「那也可以只依賴Mixin，不是不能依賴，而是耦合度要小到只有幾個接口」。以澄清為準，不禁止所有依賴。

Frontend 與 mixin 可在 program/v2 專用子模組；一般 program 只用公開 mixin 與少量契約，不直接操作 interpreter／engine internals。物理 backend 留在數值 owner，YAML 與組裝不移入 program。實作時新增 import contract 限制繞過公開入口的直接依賴，但允許入口正常委派內部；不得新增或放寬 ignore_imports。

第一版涵蓋確定性巢狀控制、ScanWith、table-driven pulse／readout 與具名 raw-word consumers。LoadValue 壓縮不改語意；ComputedPulse 保留最大候選排程；TableLengthPulse 使用實際選中長度；TablePulseReadout 保留 init、loop-after 更新及回繞。Join 不支援跨支路 register 相依。Reset 只發實際事件，不強設 ground，FTF 不新增 BathReset reservoir 物理。

Timeline 分開 pulse events 與 ADC windows，保留量化座標、signed complex samples／sample axis、carrier、phase reference、通道與來源。Frontend 不讀物理參數或選 solver 步長；backend 不解釋 Module 類別。不同控制路徑有對應 timeline，相同控制可共用不可變資料。

### 固定訊號契約與重現

2026-10-14，使用者確認：「認同，所以一個enviroment可以自由定義的部分應該是演化的方式，而輸入(timeline等)與輸出(訊號等)應該是固定格式。」Backend 接收固定 RoundRequest，一次回全部 ADC，不接收 program、driver 或 session，也不組裝最終 acquire 結果。

對 backend 求和、adapter 處理 buffers、原 acquire 正規化與平均的分工，使用者回答「認同」。Backend 回 demodulation 與 receiver 響應後的 float64 ADC sample units：integrated 為每 ADC `(execution, read, 2)` 的離散 sums；decimated 為 `(execution, read, sample, 2)` 的 traces，不平均 reps。不同 ADC 可有不同 sample count，末軸為 I、Q。

Adapter 處理 QICK channel ordering、shape 與 nearest-even int64 rounding；非有限或超界報錯，不 wrap／clip，不新增完整 ADC 飽和模型。原 acquire 處理 normalization／averaging，decimated 只平均 software rounds。Companion accumulated 由同一 trace 求和，不抽第二份 noise。Integrated 可直接算等價離散和與 noise statistics，不強制配置所有 traces。

Root seed 可指定或生成。各 acquire 使用獨立抽樣流，round／point／shot 有明確身份；取消不重用已消耗的 acquire 序號。只快取確定性計算，完整 key 與 round index／seed 分開；清 cache 不改相同輸入與 seed 的結果，不承諾跨 library 版本 bit-exact。

Acquire 與 round 記錄可引用去重，但模型、waveform、設定內容必須保留且可解析，不能只留下可被覆寫的路徑。正常 GUI／MCP 只取得 I/Q、分析、控制值與公開 provenance；private truth／重播資料分開，不以模型真值幫操作者校準。這是介面界線，不是對本機檔案權限的 sandbox。

### 第一版模型組合與控制非理想

2026-10-13，使用者對保留 arbitrary I/Q 與第一版模型組合回答：「你說目前把 arbitrary I/Q 轉成 abs(I+jQ)是目前的SimEngine嗎？確實不理想。採用建議」。FTF 在含耦合聯合空間演化，產生 coupler 調諧、混成、idle ZZ、conditional phase 與 leakage，不直接套理想 CZ。

F1／F2 使用 Fluxonium，coupler 是真正 flux-dependent circuit。使用者選擇：「1A，就是要練習用實驗找工作點」。F1–C、F2–C 與 F1–F2 皆用完整 charge coupling；2026-10-14 使用者要求「可能要考慮非零的F1-F2 coupling」。g12=0 只表示關閉該項，非模型假設。

控制包含 bias offset／gain、固定 flux 串擾、可含相位的 microwave 串擾、來源線 delay／一階低通。先來源線響應、後 loop 映射；同 shot 保留 tail，每個獨立 shot 從當下 DC 線路穩態開始。不建立任意 transfer-function framework，不丟失 envelope 符號、quadrature、carrier 或 phase。

### Coupler 固定座標與固定演化基底

2026-10-14，使用者對固定電路座標／固定低能投影回答「同意」。Coupler 採零 loop-inductance、對稱 flux 分配的有效 SQUID，不加入額外 loop modes。固定 θ 下：

```text
Hc(f)/h = 4 EC (n - ng)² - EJmax cos(πf) cosθ + d EJmax sin(πf) sinθ
f = Φ/Φ0, EJmax = EJ1 + EJ2, d = (EJ2 - EJ1)/(EJ1 + EJ2)
```

靜態可經相位平移對應 TunableTransmon effective EJ；含時演化保留固定 θ，不混用不同 flux 的 shifted-frame eigenvectors。不宣稱任意快速 flux 的電容分配／感應電壓效應都精確。

原生算符先投影至固定 local levels，再組 FTF product space；不建立三個完整 native spaces 的巨大 tensor product。每個獨立 DC 控制點由無外部 pulse、含耦合 H0 選最低 M 態形成 V。Hamiltonian、drive、dissipation 一致表示在固定 V；同 shot 不換基底或重設狀態。n² 先形成再投影，不能平方已截斷 n。

使用者裁決：「結果是不是收斂的模擬器不要保證，讓用戶自己透過yaml檔設定並自己負責，可以有註解標識建議。」Cutoffs、local levels、M、步長與容差由使用者設定，不自動收斂認證或擴張基底，不用 sink 或靜默重新正規化掩蓋失敗。M=8 不是通用預設；開發 reference 比較不成為 runtime 保證。

### 以既有耗散模型推導相干時間

2026-10-13，使用者要求：「我認為應該使用T1_curve與T2_curve裡的工具，使用Temp/Q_cap/Q_ind/x_qp等在yaml裡的參數來推導T1/T2值。」重用物理 spectral models，不新增手填 T1／T2 的平行真值。

[T1 工具](../../../lib/zcu_tools/notebook/analysis/t1_curve/README.md)使用 Temp、Q_cap、Q_ind、x_qp；[T2 工具](../../../lib/zcu_tools/notebook/analysis/t2_curve/README.md)另有 A_phi、n_th。Temp 單位為 K。有效01 equilibrium T1 不能當每條 jump 的 directional rate；echo characteristic rate 也不能當任意 Ramsey／gate 的精確模型。

數值公式由數值 owner 提供，分析與模擬共用，不由 preset 複製或 solver 呼叫 Notebook workflow。`simulate.spectra` 集中 loss spectra，`simulate.baths` 與 `simulate.fluxonium.coherence` 共用。`simulate.dephasing` 提供 T2 純計算與 CP 多能階 dephasing，Notebook 呼叫數值 owner，不新增逆向 import 或放寬 contracts。舊 T1、pointwise Q、Purcell 單位修正已獨立合併，契約見 [[0071]]，不再列為此 task 待辦，也不自動改歷史 fit／Q。新 joint kernel 仍須獨立單位、零溫與小 gap 驗證。

### 多體 directional rates 與 coupler 耗散

2026-10-14，使用者對分段更新 dissipator 與第一版 loss channels 回覆「採用」。使用 joint gaps 與 bath matrix elements 算 rates，共同 Temp 決定上下躍遷的 detailed balance。固定 H0／無 drive 時保持 retained Gibbs 穩態，不保證唯一熱化終態。

F1／F2 納入 capacitive、inductive、quasiparticle 有效模型。Coupler 以 Q_cap 為基線，有明確有效阻抗／互感參數時可加 charge-impedance 或 flux-bias-line loss；不搬用 Fluxonium inductive／qp 公式。Baths 先彼此獨立，joint mixing 不代表新增 correlated bath；控制串擾或共享 readout line 也不自動增加共同 relaxation bath／Purcell。

Dissipator 依 filtered flux 分段更新，用無 microwave、含耦合 H 建 instantaneous dressed jumps，再轉回固定 V。Coherent Hamiltonian 依實際 waveform 演化，段間不重設狀態；靜態條件可重用，不每個 microwave carrier cycle 重建。這是 weak-coupling Markov secular 近似，不含 drive-dressed／Floquet bath 或 memory。

事件區間不等於常數 H 區間。Backend 依使用者精度設定處理時變 envelope／carrier／flux；固定 generator 可重用傳播作用，時變區間重用算子並積分。段間接續量子態、filter、phase 與讀出場；進入狀態不同時不可直接重用上一段輸出。

### 有效 pure dephasing 與殘餘 photons

2026-10-14，使用者對有效 Markov dephasing 與 n_th 顯式／缺省來源回覆「都採用」。F1、coupler、F2 的 A_phi 以 joint flux sensitivity 差套用 first-order echo characteristic rate；各 loop 先獨立，不從控制串擾推造 noise covariance，不加入二階 flux noise。

Photon dephasing 使用 joint-state resonator pull 的完整差、total linewidth 與 n_th，不因參數名 chi 再除二。不同 resonator photon fluctuations 先獨立，共享線路 coherent response 仍保留。兩種 pure dephasing 在相同 flux segments 更新，須有合法多能階 generator，保持 populations／Gibbs，不重複加入 relaxation 的1/(2T1)。

每個 resonator 可明示非負 n_th；省略則用共同 Temp 與 bare frequency 的 Bose occupation，明示0或零溫缺省0關閉該項。非法值不能當省略。Driven coherent photons 不計入 n_th，不另建 thermal bath 溫度或 photon population dynamics。

這是低 occupation 下的 exponential Markov 近似，不生成精確1/f包絡、Ramsey／echo 差異或逐 shot 低頻 noise trace，echo 不消除此不可逆 dephasing。適用假設留在文件，不以事後降低 I/Q contrast 代替演化。

### 分段引擎與重算式重播

FTF acquire 預設使用 `simulate.evolution.Engine` 的快速分段算法。`experiment.ftf` 以模板陣列換算單位、取得 DC 模型，將一組 timeline 交給引擎。引擎切段、分類、去重，先建立共用 prepare，再以 joblib threading chunks 演化至 readout 起點。每個 `FTFSource` 持有一個引擎，跨 acquire 保留記憶體中的 prepare 快取。每次 evolve 自己建立及釋放 worker pool。

離線 Intel OpenCL 建表、U-table library、archive/reuse/validation 發布流程與 persistent exact density records 已移除。FTF 不需要數值資料庫路徑，也沒有 `build_database.py` 或 `gpu` extra。引擎內部的週期表與 gate prepare 只在記憶體中重用，不是磁碟建庫機制。Prepare-cache 預算不包含模型、solver、ADC、回傳 density 或仍被工作持有的資料，不是 process RAM 上限。

每 round 仍捕捉當下 snapshot。同一 acquire 內，snapshot 與上一 round 相同時，組合層重用 density 與條件響應；抽樣身份仍隨 round 改變。Snapshot 改變則重新演化，不跨獨立 shot 承接量子態或隨機輸出。

Private replay 保存模型設定、模板 request、snapshot 與抽樣身份。`replay_round(round_path)` 不讀取舊 density 或 U 表，而是在相同 backend、model、`SEGMENT_NUMERIC_VERSION`、RNG 與相依套件版本下重算。Manifest 不再釘住 record；舊 exact-record manifest 明確失敗，不靜默升級。私有輸入含模型真值，不進一般 Run/native/RPC metadata。

數值 API、prepare 快取與記憶體歸算見 [evolution README](../../../lib/zcu_tools/simulate/evolution/README.md)。早期離線建庫方案只保留於[討論快照](../../discussions/2026-10-15-offline-propagator-library.md)，不再描述現行入口。依賴方向仍由 [[0072]] 的 C8、C9、C12 保護，沒有新增 simulate→resources／program／device 依賴。

### 演化與 ADC 數值工具

數值能力位於既有 `zcu_tools.simulate` 下的子模組，不新增頂層 package。Library 提供演化與 ADC 兩個可分別使用的工具，由 environment 作者組合，不強制綁成單一使用入口。公開入口為 `simulate.evolution`、`simulate.adc` 與 `evolution.populations`。`resources.ftf` 擁有具名 YAML 與 field-path 診斷，`experiment.ftf` 組裝固定單位裝置、接線及逐 round snapshot，`simulate.ftf` 只收數值。依賴沿 C8、C9、C10、C12，不新增 ignore。

2026-10-15，使用者裁決：「你剛剛不是設計成一個子module嗎？怎麼邊package? 1235認同 4我認為可以lib可以拆成兩個工具，分別處理演化與ADC，讓enviroment作者使用」。其中 1、2、3、5 分別是提供數值模型、保留連續控制描述、統一演化入口並內部選算法，以及模型／資料可常駐但每次明確傳入初態。

模型建立方提供一致 basis 下的 Hamiltonian、drive operators 與耗散資料；演化工具不解析 FTF YAML 或 QICK Module。控制保留 envelope、frequency、phase 與絕對時間，只有真正的 arbitrary waveform 才以其 samples／時間軸表達。演化工具集中處理切點量化、聯合區段、算法與資料重用，呼叫端不逐段挑 solver。

演化輸出量子態。Environment 作者顯式呼叫 library 的 population 轉換工具，再將 population vector 與匹配的數值讀出模型、採樣設定及抽樣身份交給 ADC 工具。轉換工具明確處理 density 的表示座標、目標 basis 與狀態順序，不隱藏在 ADC 內。FTF 的條件平均讀出與 noise 契約不變，QICK buffer 格式轉換及既有 acquire 正規化仍由 program adapter 負責。

2026-10-15，使用者裁決：「同意，我預期是：basis+density matrix -> population vector -> adc iq，每步由enviroment作者調用工具達成，這樣也能方便紀錄population演化成圖（可用於debug）」。隨後澄清：「我說演化有歧義，我是只沒變成iq的population」。因此 environment 作者可以保存或繪圖檢查 I/Q 生成前的 population，不因此新增時間 trajectory 或內部積分步輸出。繪圖不放入數值工具，診斷 population 不混入量測訊號格式。

模型及 prepare 快取可重用，量子態不隱含承接上次呼叫。同一 timeline 內由引擎明確接續，跨 shot／round 不繼承。數值工具不自行猜測 round；resources 只保存私有重播輸入，不保存演化表或 density records。

依賴仍受 [[0072]] 的 C8 分層、C9 無循環及 C12 穩定工具不依賴有狀態能力模組約束。這不建立 simulate→program／resources 的新依賴，不放寬 ignore_imports；`program-simulation-entry` contract 限制一般 program 只依賴公開 mixin 入口，入口可委派 frontend 與 adapter。

### 多體演化與讀出模型

Readout 採弱讀出、dispersive、線性 resonator 與能量基底 QND mixture，不增加 resonator quantum Hilbert space。引擎演化至 readout 起點並回到 DC 能量基底。Readout 固定使用 DC 格點的 basis、rates／pulls，忽略殘餘 flux tail 及重疊 controls；起點之前的 control 部分仍正常演化。每點必須只有一個末端 readout pulse。這是讀出近似，不宣稱 tail 已消失或 ADC 支援時變 flux。二階 dispersive pulls 對所有可佔據 retained states 計算，virtual transitions 可來自較大的 parent space，不給 leakage／coupler states 任意零 pull。

2026-10-14，使用者提出：「readout_source/receiver一個就好，因為共用傳輸線。」對是否也只用一個 generator 與一個 ADC readout channel 的追問，確認：「是的，只有一個gen_ch與ro_ch」。FTF 的 readout_source 與 receiver 各一個，分別綁定 YAML 指定的 gen_ch 與 ro_ch；不合併 microwave／flux control 的其他 generators。兩 resonators 的分次換頻或同次多頻策略尚待收斂，不因本裁決擴大末端讀出時段的契約。

R1、R2 共用 reciprocal、co-located two-port transmission line，包含 off-diagonal radiative damping，不等同兩條獨立 Lorentzian 相加或僅共用 ADC。Two-port 指物理端口，不代表雙 ADC。單一 receiver 觀測共同線路輸出，接收 noise 的 PSD matrix 定義在 I,Q 空間，為2×2實對稱正半定矩陣。有限 pulse 經 classical field evolution，再依 ADC window demodulate、sample／integrate 並套 receiver 響應。

2026-10-14，使用者對「保留空間內的多能階relaxation＋條件平均讀出，不追逐shot衰減細節」回答「好，同意」。對每個 readout 初始 joint state 計算含既有 directional relaxation／thermal transitions 的條件平均響應，涵蓋矩陣元允許的 cascade、直接跨級、coupler／leakage states 與熱激發，不限0↔1。不新增 readout T1、空間外態或 ground sink。

響應保留 population 與 state-conditioned 場 moments 的關聯，不能只平均 resonator frequency 再解單一場方程。每 shot 從演化後的 populations 抽一次 readout 初態，單一 receiver 使用該 joint 初態下兩 resonators 的共同平均響應，再加入 I/Q receiver noise；raw／averaged 共用資料。這不同於每 shot 抽 thermal pure state 再重解量子演化。

此模型保留初態關聯與平均對比，不生成跳變時間造成的尾巴、額外 variance 或跨通道隨機關聯，不用來驗精細 single-shot fidelity。不保留逐 shot 軌跡切換模式，不新增 power-induced transitions、額外 readout Purcell、強 backaction 或 Kerr。相同物理條件可共用演化與條件平均響應，reps／rounds 主要增加抽樣及資料輸出，成本仍須實測。

### Lookback、適用域與 GUI 邊界

2026-10-14，使用者裁決「1納入 2你跳變的計算難度為什麼與reps/rounds有關？ 3不檢查也不管 4同意 5可以」。其含義為納入物理 lookback、不介入近似適用域、接受 seed 重現方向，以及允許 YAML 草案；不代表批准完整 schema 或新的 benchmark。

Lookback／acquire_decimated 使用相同有限 pulse 與條件平均場，輸出時域 ADC 資料，不以 TLS／白雜訊替代，也不聲稱生成單 trace 的隨機衰減跳變。不同 acquire 使用各自抽樣流。

不建立 mixing、n/ncrit、短窗、背景 photon occupation 的 runtime 適用域門檻、警告、拒絕或 quality policy。模型假設仍明示，不代表任意設定都準確。必要輸入合法性、未支援操作與真正 solver 失敗仍報錯，非有限或空資料不能冒充成功。

同日，使用者要求：「我不希望改動太多gui內的邏輯，前者不要有太多模擬認知」。GUI 只做必要組裝與既有流程接線，不新增 T1／物理判斷、專用警告面板或 Run 狀態。一般錯誤沿通用契約傳遞，通用能力不足時回報缺口，不另建模擬機制。

### White_noise preset

2026-10-15，對「零均值、獨立 Gaussian＋單一 std_adc 幅度參數」的提案，使用者回答：「可以，下一步要做什麼？以及你有將討論與決策寫入discussion/adr嗎？」White_noise 不含量子演化，不對 pulse gain、frequency 或 device bias 產生物理響應。std_adc 是 demodulation 後每個離散 ADC sample 的 I、Q 標準差，單位為 ADC sample units，有限且非負；零值明確關閉 noise。I/Q、channels 與獨立樣本互相獨立。

對含 N 個 samples 的窗口，decimated 的每 quadrature sample 為 N(0,std_adc²)；integrated 回 sample sum，可直接抽等價的 N(0,N×std_adc²)。這些分布定義在 adapter rounding 前；int64 rounding、normalization 與 averaging 仍走共用 acquisition。Companion raw 從同一 decimated trace 求和，不另抽 noise，raw／averaged 共用資料。

此 preset 遵守共同 seed、round identity、重播與清理契約，不使用全域 RNG，不作任何失敗 fallback。不新增非零 mean、相關 noise 或物理模型；公開入口是 `experiment.simulation.white_noise(std_adc=1.0, seed=None, ...)`，回傳相同的具名環境 bundle。

### 舊 TLS 與 Notebook 遷移

2026-10-14，使用者裁決：「我不想留舊TLS機制，notebook可以相應的遷移，白雜訊mock也可以考慮移除，還有什麼是舊frontend」。舊 TLS 執行機制退役，Notebook 遷移到新環境，不保留 SimParams／SimEngine 相容層。舊模擬 frontend 是 Module 到 TLS/Bloch 的轉譯，不是 GUI 或正式 QICK 編譯器；可參考解碼知識，但不要求保留舊實作或內部 interface。

2026-10-16，對移除舊隱式白雜訊及雙模式 mock API、不留相容層的建議，使用者確認：「可以，請定稿spec與拆tickets，但不要實作，我會交給獨立agent實作」。移除舊隱式 noise acquisition 與 make_mock_soc／MockQickSoc 的雙模式公開契約，white_noise 只由新環境明確選用，不作失敗 fallback。這避免同時維護隱式模式與新的環境生命週期。

保留參數化 make_mock_soccfg 的離線設定／編譯能力，QICK adapter 仍歸 program。舊類別可重用的內部邏輯可移到新入口，但不保留舊雙模式 API。Framework fixtures 按被測 seam 使用合成 executor，不一律執行 FTF。只服務已刪邏輯的 tests 一併移除，不新增「舊內容不存在」的斷言。

### 分工契約與模組邊界

2026-10-16，使用者要求：「我反而認為應該是先完成每個分工模塊的contract，然後各自內部平行開始，因為各自都是黑盒」。對六個契約與先定契約的流程，使用者回覆「確認」。

依 [[0072]]，`core` 放無狀態型別，`simulate` 不 import `program`，`experiment.ftf` 負責組合與單位換算。依賴方向由 C8 分層、C9 無循環、C12 穩定工具不依賴有狀態能力模組保護。一般 program 只依賴公開 mixin 入口，由 `program-simulation-entry` contract 限制，不繞過入口操作 frontend 或引擎。

| 契約 | 生產者 | 消費者與邊界 |
| --- | --- | --- |
| C1 模板請求 | `program.v2.simulation` | `experiment.ftf` 消費 `core.simulation` 的模板請求，不把 Module 物件交給數值引擎。 |
| C2 round 結果 | `experiment.ftf` | program mixin 消費 `core.simulation` 的統計或訊號結果，負責 QICK buffer 轉換。 |
| C3 引擎 API | `simulate.evolution` | `experiment.ftf` 呼叫演化入口，消費 readout 起點的量子態與路徑診斷。 |
| C4 模型 | `simulate.ftf` | `simulate.evolution` 消費模型與 secular 形式，不讀 YAML 或裝置。 |
| C5 kernel | `simulate.evolution` 內部的引擎排程與各 kernel | 排程提供分段工作與預算，kernel 提供 prepare、evolve-local input-bound payload 與 apply。A-stage 的共用 prepared batch 可透過 bind_input 接收同 key、同 port、起點為零的 rows 與 row basis 中的 source 初態。唯讀 payload 只活在單次 evolve，不進 PrepareCache。回傳 None 時保留 per-batch apply。這是引擎內部接縫，不是跨 package API。 |
| C6 readout 與統計 | `simulate.adc` | `experiment.ftf` 消費條件響應與統計抽樣工具，將 population 組合成 C2 結果。 |

### Secular 形式由模型提供

2026-10-16，對模型直接提供 secular 形式、模型不提供就 fast fail 的方案，使用者選擇「2A」。

模型提供 rate matrix Q、coherence 衰減 Γ 與 H0 eigenbasis。快速路徑直接消費這些量，不在執行期從 collapse operators 反推 secular 結構。模型未提供時，在進入點報錯，不保留驗證失敗後退回另一條演化路徑的分支。

### 快速分段算法是預設數值方法

2026-10-16，對快速算法直接取代模擬環境預設數值方法的提議，使用者回覆：「同意，這個task就是來優化算法的」。

模擬環境預設使用快速分段算法，不保留另一條精確演化路徑。分段引擎內部依條件選用解析解、近似或積分 kernel，不由呼叫端維護快速／精確雙模式。嚴格 reference 用於開發驗證，不是另一個 runtime backend。

## 交付狀態與限制

- Program 提供具名環境 binding、Module timeline 與 request／signal 契約。GUI 先發布同一 manager 的 live devices，再還原 remembered devices。正常硬體路徑不變，模擬的 raw compile hooks 明確拒絕。
- Resources 提供 FTF YAML schema 與 checked JSON private replay 容器。Experiment 組合環境及單 round 重播；重播用相同數值與 RNG 版本重算，不引用已保存的 density。
- Evolution 提供預設快速分段算法及記憶體 prepare 快取。離線 U 表、Intel OpenCL、持久 exact records 及建庫入口已刪除。快取預算不是 process RAM 上限。
- 舊 TLS、隱式白雜訊及雙模式 mock API 已移除，Notebook 使用新環境。沒有相容層。參數化 make_mock_soccfg 仍支援離線編譯。

I/Q-only 練習在原版本已跑通 F1、F2，conditional Ramsey 只取得一點。驗收採「能跑起來」，不代表全 16 點完成或原精度達標。歷史正式 workload 的完整 acquire 成本為 cold 平均 2.864 秒／point、warm direct-record 平均 0.457 秒／point，不能稱平均 1 秒已達標。這些數字不能代表現行分段引擎。

現行 FTF backend 是 `ftf-segment-v4`，使用分段引擎與重算式 replay。舊版本 private archive 透過公開 replay 入口明確報版本不符，不靜默升級。原校準、raw、seal 與成本證據維持原版本；此處不把舊證據當作目前版本的科學精度或效能驗收。

固定 original basis 的內部自洽驗證與跨 basis population 比較仍保留。跨 basis phase／complex 偏差照實保留，不作通過條件。模擬的絕對頻率與相位有 MHz 級 basis 偏差，不適合對照外部能譜或硬體絕對值。這些限制不改變既定物理模型，也不構成 runtime 收斂認證。
