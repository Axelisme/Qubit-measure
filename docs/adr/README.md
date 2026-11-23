# Architecture Decision Records

**讀者：** framework-dev、agent。透過本索引與各篇 ADR 查找架構決策及其依據。

依 [CLAUDE.md〈使用者決策的引用〉](../../CLAUDE.md#使用者決策的引用)，取捨符合任一條就寫進 ADR：對多個底層模組已有或可能產生約束；定下整個子系統共用的架構；不易撤回，或得記下為何否決其他方案。像 Python 的 [PEP 659](https://peps.python.org/pep-0659/) 一樣，內部的重大架構改變也收錄，小改進不收。Evolution 的 kernel 共同規則留在 [ADR-0078](0078-evolution-module-boundary.md#每個-kernel-的量測場景只有一段)，個別 kernel 的閾值寫在 [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md#kernel-summary)。只有單一實驗自己用、能在一個 task 內反悔的取捨，寫在該實驗的 README。獨立 heralded 入口的取捨則寫在管理這些實驗入口的 [zcu_lab/v2/README.md〈何時保留獨立 heralded 入口〉](../../zcu_lab/v2/README.md#何時保留獨立-heralded-入口)。索引只放 ADR 原則的摘要與指標。
ADR 正文寫目前生效的規則與客觀事實。裁決日期、使用者原話與沿革的放置方式見下方[格式規定](#格式規定)。依據中的「使用者未覆核」只標尚未覆核的代裁；已覆核時記下覆核結果。

程式碼註解與記憶檔以 `ADR-NNNN` 引用本目錄；ADR 之間以 `[[NNNN]]` 互鏈。

有編號的正文標題使用 `NNNN — 標題`，無編號的草稿與退役提案保留原標題。狀態只記在 YAML front matter；日期有原文來源才填，不用入庫日期代替裁決日。

ADR 宣告 package 之間的依賴方向時，同時在 `.importlinter` 建立對應 contract，並在本文寫出 contract ID。沒有 contract 的依賴規則只能靠判斷維持，視為 ADR 未完成。

## 格式規定

每個決策節依序寫四段：

- **規則。** 用現在式寫目前生效的 owner、行為與責任邊界。
- **例子。** 一個能判斷規則怎麼用的例子。
- **邊界。** 適用範圍，以及什麼條件下改選另一種做法。
- **依據。** 裁決日期、使用者的「」原話與討論紀錄連結。代裁寫明「designer 代裁，使用者未覆核」；原始討論未入庫時註明。

被取代的規則與選項比較不留在正文：放討論紀錄，或靠 git 追溯。已核准、尚未實作的方向放 `draft/`。節名保持穩定；被其他文件引用的節要改名時，先更新所有引用。

## Module Boundaries

原則：組合根與開發工具只經 assembly 依賴使用者 lab，固定量表場景則不依賴 lab，見 [ADR-0072](0072-module-boundaries.md#使用者空間的依賴原則)。
原則：模擬器（`simulate`）是封閉的子模塊，不 import `zcu_tools` 其他 package；具體裝置模型由使用方經聲明介面提供，見 [ADR-0072](0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明)。
原則：拿掉 `SimulationAcquireMixin`，Program 不留任何模擬邏輯、參數或型別，見 [ADR-0072〈真實 acquire 與模擬邊界〉](0072-module-boundaries.md#真實-acquire-與模擬邊界)。

- [0072 — 模組定位與依賴邊界](0072-module-boundaries.md)：Package 定位、依賴方向、各實驗獨立取數與 analysis 小摘要契約，依賴方向由 `.importlinter` 檢查。

## Concurrency / Lifecycle

- [0001 — Domain guard 與 Lease 分離](0001-permit-lease-typed-guard.md)：Domain guard 與 lease 的責任分界；typed Permit proof 已由 [ADR-0066](0066-operation-lifecycle.md#問題與決策) 取代。
- [0002 — Version table + off-main handler](0002-version-table-async-handle-off-main.md)：資源版本表與 RPC off-main handler；handle 舊模型已由 [ADR-0066](0066-operation-lifecycle.md#問題與決策) 取代。
- [0053 — Owner scheduler 與 gate presence](0053-owner-scheduler-and-gate-presence.md)：Owner scheduler、State guard、completion facts 與 gate presence。
- [0066 — Operation 執行、取消與關閉](0066-operation-lifecycle.md)：Operation admission、lease、取消、完成、shutdown 與 disconnect 的責任分界。

## GUI Service Architecture

- [0064 — GUI process startup 與 app composition](0064-process-startup.md)：Launcher、runtime、composition 與 startup／on_close 的資源和 session 交接。
- [0067 — GUI 應用核心與前端邊界](0067-gui-application.md)：Shared session、應用核心、前端反應與各 GUI pane 的 owner。
- [0076 — Device 活動快照與 run 後讀回](0076-device-activity-and-readback.md)：Driver 活動狀態、DeviceManager 快照、單名稱 registry 與 run 後 cache 讀回。局部規則見 [FakeDevice 的 address 身分](../../lib/zcu_tools/device/README.md#fakedevice-的-address-身分)與[裝置活動標記](../../lib/zcu_tools/gui/session/ui/README.md#裝置活動標記)。

## Cfg / Value Model

原則：表單預設引用 library，只在改了 Run 不會覆寫的欄位時複製，見 [ADR-0073](0073-measure-form-derivation.md#表單預設引用-library只在改了-run-不覆寫的欄位時複製)。
原則：掃描初值每軸至少 31 點、以中心為準的用奇數；只管初值，不擋輸入，見 [ADR-0073〈Form 的形狀與初值〉](0073-measure-form-derivation.md#掃描初值的點數)。
原則：表單初值用簡單固定的公式，不依板子調整；可填算好的數值或寫 expression，兩種寫法都遵守與板子無關的固定範圍，見 [ADR-0073](0073-measure-form-derivation.md#自動推導點到為止)。
原則：初值在一般 point 上就違反實驗自己的規則時，修初值公式，不加檢查；依板子的限制不改初值，見 [ADR-0073](0073-measure-form-derivation.md#預設值的瑕疵就改預設值)。
原則：依板子而定的限制先不加程式檢查，由 Run 報錯，判讀放知識庫的 judge 條目，見 [ADR-0073](0073-measure-form-derivation.md#板子限制先不加程式檢查)。
原則：表單只用現成機制做簡單檢查，例如單一欄位 annotation 的固定範圍；需要新增並長期維護的機制不做，交給 Run 或知識庫；防止默默產生錯誤資料的守門不受此限，見 [ADR-0073](0073-measure-form-derivation.md#表單只做現成機制做得到的簡單檢查)。

- [0008 — Headless Form 與 writeback draft](0008-cfg-editor-session.md)：Headless Form 與 writeback draft 的宿主所有權及 Apply 提交邊界。
- [0037 — Session value lookup](0037-measure-gui-value-lookup-resolve-once.md)：唯讀 value lookup、resolve-once 與固定 capture。
- [0065 — Cfg 編輯模型與使用邊界](0065-cfg-editing.md)：共用 Form、持久 Spec、固定來源與 Library／Inspect／writeback 提交邊界；欄位轉換與草稿還原見 [ADR-0079](0079-form-field-transitions.md#決策狀態與問題)。
- [0073 — Measure 表單從 core cfg 推導](0073-measure-form-derivation.md)：Measure Form 從 core cfg 推導、宣告初值 factory 與實驗 record 的邊界；欄位轉換與草稿還原見 [ADR-0079](0079-form-field-transitions.md#決策狀態與問題)。
- [0079 — Form 欄位轉換、固定元素與逐元素操作](0079-form-field-transitions.md)：已採用但尚未完成實作的 Form converter、scoped command 與固定元素設計。

## Remote / Transport

原則：recipe 與 workflow 對 agent 的通用契約只有 `ask` 與正常結束，其上的工具都是 helper；helper 不好用時修 helper，不擴充原語，見 [ADR-0075](0075-agent-interface-layers.md#通用契約只有-ask-與正常結束)。

原則：便利性的小功能要等同一個問題多次出現才加，錯誤結果、資料遺失或擋住流程的問題第一次就修，見 [ADR-0075](0075-agent-interface-layers.md#先做簡單設計再按需加法)。

原則：guide 寫給懂物理、沒看過實作的讀者，只介紹實驗與它的行為；板子限制與底層報錯的判讀放知識庫的 judge 條目，見 [ADR-0075](0075-agent-interface-layers.md#判讀依據放知識庫)。

- [0068 — 遠端前端、傳輸與 agent 介面](0068-remote-transport.md)：共用傳輸、holder dispatch、RPC／MCP 與事件錯誤投影。局部等待策略見 [Fluxdep Auto Align／search](../../lib/zcu_tools/gui/app/fluxdep/README.md#auto-align-與-search-的等待)與 [Measure MCP `device_setup`](../../lib/zcu_tools/mcp/measure/README.md#device-setup-的等待)。
- [0070 — GUI 與 agent 的控制身份](0070-control-identity.md)：Measure／fluxdep 單一 holder、固定連線身份、控制權交接與 GUI 寫入保護。Measure 的局部呈現見[錯誤通知](../../lib/zcu_tools/gui/app/measure/ui/README.md#錯誤通知)。
- [0075 — Agent 介面分層](0075-agent-interface-layers.md)：Agent 介面分層、共用 async host、顯式 Scope 與跨層 writeback。局部規則見 [Fluxdep 的流程依賴](../../lib/zcu_tools/gui/app/fluxdep/README.md#fluxdep-的依賴模型不是跨-app-原則)。

## Persistence

原則：同一種資料不因屬於 global 或 point 分成兩套型別或方法，選檔用 `point: str | None`（None 是 global），見 [ADR-0063](0063-persistence-ownership.md#同一種資料不因-global-或-point-分兩套)。

- [0063 — Persistence 保存權威與資料契約](0063-persistence-ownership.md)：保存權威、Container 讀寫、Run metadata、library 與 component 升級邊界。
- [0080 — datafile 通用容器格式與獨立介面](0080-datafile-generic-container.md)：只接受 Container major 3 的通用 vars／docs／groups 容器；公開單一 Variable 的無 I/O 結構檢查。
- [0081 — Global 與 point 資料以磁碟為真相](0081-point-data-disk-sync.md)：Global／point 以磁碟為真相、write-through、watcher 與 revision 樂觀鎖。
- [0082 — Component flux frame 與明確升級](0082-component-flux-frame.md)：Component offset／signed periods、flux_dev 與明確 v1→v2→v3 升級鏈。

## Experiment runtime / Readout / Workflows

原則：一般實驗由取數端加總或平均已完成 rounds，Result 不留 round 軸；這是建議，需要逐輪資料時可把 round 迴圈放在最外層，見 [ADR-0083](0083-program-macros-and-timelines.md#round-在取數端歸約)。

原則：實驗 cfg 可為統一命名改欄位；改名升 schema major，不遷移、不留 alias，舊 run 在結果仍相容時保留 Result 與 metadata，見 [ADR-0069](0069-experiment-core-and-frontends.md#cfg-欄位改名與-schema-版本)。

原則：adapter 解析元件身分，cfg 收位置數值，分析 options 收顯示名。同一位置不跨層翻譯，實驗只驗所需能力，見 [ADR-0069](0069-experiment-core-and-frontends.md#身分定位與顯示)。

原則：懂物理的操作員不讀程式碼就能判讀實驗。圖只放判讀所需的事實，預選理由留在 Writeback；adapter 提供 GUI 與 MCP 共用摘要。見 [ADR-0069〈呈現規範〉](0069-experiment-core-and-frontends.md#呈現規範)。可從同批統計量推導的棒都畫，含一階傳遞；Live 也畫，見[能推導的誤差棒都畫](0069-experiment-core-and-frontends.md#能推導的誤差棒都畫)。

原則：報告只寫本次做了什麼；沒做的步驟留 null、原因記 log，程式讀者需要區分原因時才加狀態欄，見 [ADR-0069](0069-experiment-core-and-frontends.md#報告只寫本次做了什麼)。

原則：Measure 圖磚以 12×9 in 排版，只等比縮放整張圖；R7 在存檔 PNG 檢查版面，見 [ADR-0069](0069-experiment-core-and-frontends.md#圖以固定存檔幾何排版)。
原則：平均 IQ 的一維圖預設用主軸投影，二維 flux／頻率圖用逐 bias 扣背景後的幅值，改用其他量須寫明理由，見 [ADR-0069](0069-experiment-core-and-frontends.md#平均-iq-的投影慣例)。

原則：Timeline 編譯器推算每發 ADC 讀取數，每個 Branch 出口與作者 label 都逐 ADC 對齊，見 [ADR-0083](0083-program-macros-and-timelines.md#每發-adc-讀取數)。
原則：Timeline program 提供通用跳轉，builder 只轉交同名動作。只往後跳，迭代用 Repeat 或 sweep。見 [ADR-0083](0083-program-macros-and-timelines.md#往後跳轉與-label-合併)。
原則：增加的量測能力不抵第二條路徑的可讀性成本時，保留一條路徑，超限就報錯。見 [ADR-0083](0083-program-macros-and-timelines.md#量測能力不抵可讀性成本時不加分支)。
原則：模擬的零輪中斷由 Program 的 `EarlyStopMixin` 統一判定；simulation mixin 只回報中斷，見 [ADR-0083](0083-program-macros-and-timelines.md#compiled-snapshot-與-simulation-入口)。

- [0062 — 實驗執行與 workflow 編排](0062-experiment-workflow.md)：Async workflow、operation、Ask、停止、live tunables 與作者保存責任。
- [0069 — 實驗核心、前端包裝與具名圖形產物](0069-experiment-core-and-frontends.md)：實驗 core 黑箱、前端包裝、Run／AnalysisRecord 與具名圖形產物。局部規則見 [`notebook/adapter.py`](../../lib/zcu_tools/notebook/adapter.py) 的「Notebook interactive analysis」註解。
- [0074 — 獨立的平鋪 Readout module](0074-flat-readout-module.md)：Readout 配置、classifier、逐 shot sums、選用校正 IQ 矩、herald reset 與逐 shot 遮罩、雙 qubit heralded 分類、單 qubit cov 自校準與 QICK projection。
- [0083 — Program macro 與具名 timeline](0083-program-macros-and-timelines.md)：Program macro、具名 timeline、往後跳轉與 label 合併、查表掃描、compiled snapshot 與全面 Timeline 遷移。

## Analysis / Simulation / Waveform

原則：模型設定只接受等向接收雜訊的單一 PSD，見[等向接收雜訊](draft/simulation-environments.md#等向接收雜訊)。
原則：family 的完整序列最終精度合格就只比耗時，刪除前先補同型態長序列證據，見 [ADR-0078](0078-evolution-module-boundary.md#family-去留合格只比耗時)。
原則：模擬器分層測試，不做 e2e 測試。只允許一支量手選場景的整實驗 Run 牆鐘工具，kernel／family 比較仍只量 `Engine.evolve`，見 [ADR-0078](0078-evolution-module-boundary.md#模擬器的測試與量測分層不做-e2e)。
原則：掃描格點及有 acquire 的逐次評估，模擬核心耗時平均不超過 0.1 秒。無軸 shots 仍用整次 180 秒，外推另報，見 [ADR-0078〈模擬器速度目標〉](0078-evolution-module-boundary.md#模擬器速度目標)。
原則：先守速度目標，誤差在 `1e-2` 內選快的；只為守時間才接受超過並記錄，達 `1e-1` 不採用，見 [ADR-0078〈歷史控制與精度驗收〉](0078-evolution-module-boundary.md#歷史控制與精度驗收)。
原則：優化只量目標實驗前後，先做便宜的正確性對照，再補獨立參考；影響實機、存檔或使用者結果則先驗再落地，見 [ADR-0078](0078-evolution-module-boundary.md#先求效率正確性事後補)。
原則：評估算法時預算固定為使用者按場景宣告的需求，在固定預算下找算法適合的場景，不為算法放寬預算；長序列預算按 gate 數放寬，見 [ADR-0078](0078-evolution-module-boundary.md#固定預算找適合的場景)。
原則：模擬算法與機制同時適用 Windows 與 Linux，不依賴單一作業系統的設定或行為，見 [ADR-0078](0078-evolution-module-boundary.md#跨平台不綁作業系統)。
原則：GPU 是可選功能，沒有 GPU 套件或可用裝置時維持 CPU 行為與逐位相同的結果，見 [ADR-0078](0078-evolution-module-boundary.md#gpu-是可選功能)。
原則：有誤差問題的快速算法先查原因並修正，修不好的範圍才收 guard，見 [ADR-0078](0078-evolution-module-boundary.md#快速算法先修好不輕易放棄)。
原則：支援範圍由物理與介面決定，不看使用頻率；頻率只排優化先後，見 [ADR-0078](0078-evolution-module-boundary.md#支援範圍不看使用頻率)。
原則：模擬器只拒絕實機也會拒絕的程式；實驗參數不當或程式合法但不合理時照常給結果並警告，契約外輸入與模擬器未實作的情況照舊 fast fail，見 [ADR-0078](0078-evolution-module-boundary.md#模擬器不替使用者擋實驗參數)。
原則：模擬器的選路、精度與維度不依程式內容被辨認成哪種實驗或架構切換；精度由使用者宣告，數值指標可以決定選路，見 [ADR-0078](0078-evolution-module-boundary.md#模擬器不從程式內容推斷實驗類型或硬體架構)。
原則：每個模擬器提供同一組調試介面，定位與驗證工具只依賴它；共通部分有型別，模擬器特有的值放 dict，見 [ADR-0078](0078-evolution-module-boundary.md#模擬器帶通用調試介面)。
原則：模擬器優化以使用者看得到的結果正確性為準，存檔格式、版本與相容性配合優化；除錯資訊寫進 debug capture manifest，見 [ADR-0078](0078-evolution-module-boundary.md#優化優先於周邊約定除錯靠-dump不靠重算)。
原則：預設選路可以讓快場景多付一筆有上限的固定成本，換慢場景大幅加速；沒被採用時超過 main 的 1.2 倍才回報，見 [ADR-0078](0078-evolution-module-boundary.md#用快場景的小幅變慢換慢場景的大幅加速)。
原則：kernel 的量測場景只有一段，重複就是同一段接 N 次，見 [ADR-0078](0078-evolution-module-boundary.md#每個-kernel-的量測場景只有一段)。
原則：場景照約化參數軸宣告覆蓋，不為個別實驗加專屬場景，見 [ADR-0078](0078-evolution-module-boundary.md#場景覆蓋靠約化的參數空間)。
原則：量測 agent 以量測取得初值與裁定；模擬器真值只在使用者為 debug 明確授權時讀，數字記為 debug 對照，見 [模擬環境草稿](draft/simulation-environments.md#固定訊號契約與重現)。
原則：ADC 擁有 `t−τ` 的 ADC 時間至裝置時間的換算；planner 逐窗以裝置時間和 block interval 配對，最後相交 block 為 owner。grid 與投影時間不移動。同 ADC、同 owner 的有場事件使用各窗交集 blocks 的 pulse 聯集（designer 代裁，使用者未覆核），見 [ADR-0078](0078-evolution-module-boundary.md#ftf-讀出窗依裝置時間配對)。
原則：FTF Stop 採合作式檢查點，正常參數下以相鄰檢查區段 5 秒內作放點依據，不設硬上限或延遲 gate，見 [ADR-0078](0078-evolution-module-boundary.md#ftf-stop-用合作式檢查點設計上-5-秒內)。
原則（designer 代裁，使用者未覆核）：FTF 只有在當輪需要 raw 時才借用取消查詢；SoC 驗證 raw 並通過最後查詢後才入帳、發佈和前移輪次，見 [ADR-0078](0078-evolution-module-boundary.md#integrated-raw-由環境生成介面由-program-擁有)與 [ADR-0083](0083-program-macros-and-timelines.md#compiled-snapshot-與-simulation-入口)。

- [0085 — 實驗數值工具的 fitting owner](0085-numerical-fitting-owners.md)：六組純數值工具由 fitting 持有，lab 保留選資料、herald、維度換算、live 與畫圖。

- [0071 — Coherence 頻率與時間單位](0071-coherence-units.md)：頻率、時間、loss parameters 與 lifetime 的單位和換算契約。
- [0078 — Evolution 黑盒邊界與算法內聚](0078-evolution-module-boundary.md)：Evolution 的 dispatch／kernel 分工、公開黑箱、[數值 preparation 壽命與 FTF 探針例外](0078-evolution-module-boundary.md#損耗漂移下的結構與-preparation-lifetime)。局部選路見 [t6 只保留 direct](../../lib/zcu_tools/simulate/evolution/README.md#t6-只保留-direct)。
- [0084 — 模擬 request 的公共整數時基](0084-simulation-integer-time-base.md)：公共有理數時基、整數 request 與 compiled frontend。固定測試數值見 [Mock clock example](../../lib/zcu_tools/program/v2/simulation/README.md#mock-clock-example)。

改模擬器時的流程與自查見 [模擬器維護規則](../../lib/zcu_tools/simulate/maintenance.md)。

局部分析／預測契約見 [fluxdep](../../lib/zcu_tools/analysis/fluxdep/README.md)、[fitting](../../lib/zcu_tools/analysis/fitting/README.md) 與 [prediction](../../lib/zcu_tools/simulate/fluxonium/README.md)；waveform 資產契約見 [repository 參考](../../lib/zcu_tools/resources/waveform_assets.md)。

## Plotting

圖形操作與更新契約見 [plotting README](../../lib/zcu_tools/plotting/README.md)，跨模組 owner 分界見 [[0072]]、[[0067]]。

## Agents

協作流程歸外部 dev-flow／collab skills，repo 的 live resource 限制見 [CLAUDE.md](../../CLAUDE.md)。

## Draft

- [模擬器獨立成 package](draft/simulator-standalone-package.md)：使用者預計把模擬器拆成 `zcu_tools` 依賴的獨立 package；目前只做到封閉子模塊，拆分未排。

- [模擬環境、FTF preset 與參數檔](draft/simulation-environments.md)：模擬環境與 FTF preset 的草稿及歷史驗證範圍，compiler owner 見 [[0083]]。
- [GUI 事件收合](draft/remote-event-coalescing.md)：GUI 訂閱端依 payload type 與 tab 的 per-tick 收合尚未落實；cfg 表單 snapshot 收合已實作。
- [Operation 關閉與 disconnect](draft/operation-lifecycle-boundaries.md)：shutdown 期限、無 handle 背景工作與 GUI device/factory owner 的未落實目標。
- [Timeline 控制結構的完備性](draft/control-flow-completeness.md)：迴圈的 continue／break 只留設計、不實作。Branch 是否改寫成跳轉未定。已實作的跳轉規則見 [[0083]]。

## Retired

退役正文保留歷史設計；現行規則見各條目的 owner ADR 或 module README。

- [0003 — Shutdown coordinator](retired/0003-shutdown-coordinator-and-registry-cancel.md)：取消與 shutdown 跨 owner 分界由 0066 承接；自動逾時關窗敘述不再是核准的共用預設。
- [0004 — Service dependency three questions](retired/0004-service-dependency-three-questions.md)：依賴語意與方向由 0067 接替。
- [0005 — Service roles](retired/0005-service-roles-ddd-hexagonal.md)：前端與 owner 角色由 0067 接替。
- [0006 — Single ml/md write authority](retired/0006-single-ml-md-write-authority.md)：寫入權威由 0067 接替；cfg lowering 見 0065 與 measure README。
- [0007 — Device state lives in State](retired/0007-device-state-to-state-ssot.md)：device 與 persistence 分工由 0067、0063 接替。
- [0009 — Spec/Value fluent + LiteralSpec lock](retired/0009-spec-value-fluent-and-literal-lock.md)：已被 [[0065]]、[[0073]] 的 Form／持久 Spec 取代，保留歷史設計。
- [0010 — Complete value tree](retired/0010-value-tree-complete-none-for-empty.md)：已被 [[0065]] 的 Form input／result 取代，保留歷史設計。
- [0011 — Finished-cfg validation](retired/0011-cfgschema-validate-boundary.md)：已被 [[0065]]、[[0073]] 的 Form result 與 adapter 邊界取代。
- [0012 — Context-free measure definition](retired/0012-cfgbuilder-value-layer-fluent-assembly.md)：舊 definition／seed 契約，由 [[0073]] 取代。
- [0013 — Remote adapter second view](retired/0013-remote-adapter-as-second-view.md)：跨模組邊界由 [[0068]] 接替。
- [0014 — Shared GUI transport](retired/0014-gui-shared-transport-layer.md)：跨模組邊界由 [[0068]] 接替。
- [0015 — GUI memento caretaker](retired/0015-persistence-caretaker-memento-single-file.md)：app 與 shared caretaker 的責任由 0063 接替。
- [0016 — notebook liveplot auto close](retired/0016-notebook-liveplot-auto-close-default.md)：notebook liveplot 預設 auto-close，不依賴 ipympl 私有協議。
- [0017 — Worker-thread plotting](retired/0017-worker-thread-plotting.md)：前端繪圖邊界由 0067 接替，機制見 gui plotting README。
- [0018 — Autofluxdep resolver builder](retired/0018-autofluxdep-orchestrator-requirement-resolver-builder-currying.md)：autofluxdep 已移除，不再提供 orchestrator 與 predictor 載入流程。
- [0019 — Operation facets](retired/0019-operation-facets-and-execution-strategy.md)：facets 與 execution 分界由 0066 承接，局部策略見 session／app README。
- [0020 — Shared session core](retired/0020-session-core-shared-layer.md)：共用／app-local 邊界由 0067 接替，局部契約見 session README。
- [0021 — Event ownership domain modules](retired/0021-event-ownership-domain-modules.md)：domain 事件與前端投影由 0067 接替。
- [0022 — Worktree coordination](retired/0022-agent-coordination-worktree.md)：舊協作 protocol；通用流程改由外部 skill 管理，本地 live resource 限制見 CLAUDE.md。
- [0023 — Cooperative interrupt feedback](retired/0023-cooperative-interrupt-feedback-wakeup.md)：舊 FeedbackInbox 由 0066 的有序 channel 取代。
- [0024 — Agent launch UI retirement](retired/0024-embedded-agent-session-architecture.md)：外部 agent 啟動邊界由 0068 接替；operation feedback 見 0066。
- [0025 — Cross-thread interaction channel](retired/0025-cross-thread-interaction-channel.md)：互動分界由 0066 承接；局部事件契約見 session／app README。
- [0026 — OperationRunner and ports](retired/0026-operation-abstraction-runner-scope-ports.md)：runner 分界由 0066 承接；局部 port 與 scope 見 session／app README。
- [0027 — Experiment data persistence](retired/0027-experiment-data-persistence-native-labber-axes-list.md)：資料責任由 0063 接替，細節在 datafile 與 experiment owner 文件。
- [0028 — Fluxdep analysis kernel](retired/0028-fluxdep-analysis-kernel.md)：flux-dependence analysis kernel 位於 GUI / notebook adapter 之外。
- [0029 — Fluxonium prediction engine](retired/0029-fluxonium-prediction-engine.md)：Fluxonium prediction policy 位於 `simulate.fluxonium`。
- [0030 — Arbitrary waveform optional recipe](retired/0030-arbitrary-waveform-asset-optional-recipe.md)：arbitrary waveform asset 是 qubit-scoped `.npz`，可內嵌 formula recipe。
- [0031 — Formula recipe segments](retired/0031-formula-recipe-complex-segments.md)：formula recipe 使用 ordered segments 與 complex expression。
- [0032 — Waveform reference time axis](retired/0032-arbitrary-waveform-reference-time-axis.md)：時間權威由 0063 接替。
- [0033 — Waveform reference lifecycle](retired/0033-arbitrary-waveform-delete-no-reference-scan.md)：不級聯更新由 0063 接替。
- [0034 — ArbWaveformDatabase repository](retired/0034-arb-waveform-database-shared-asset-repository.md)：資產操作現由 `resources.ArbWaveformDatabase` 擁有。
- [0035 — MCP failures use tool errors](retired/0035-arb-waveform-mcp-failures-use-tool-errors.md)：arb waveform error 由 main app 投影到 remote 契約。
- [0036 — Adapter capability contract](retired/0036-adapter-capability-contract-validated-at-import.md)：跨 owner 規則由 0067 接替，精確 hook 見 experiment adapter README。
- [0038 — Executor ResultTree](retired/0038-executor-result-tree.md)：共用 ResultTree 與執行骨架見 `lib/zcu_tools/experiment/v2/runtime/README.md`，workflow collection 邊界見 0062。
- [0039 — QubitParams JSON owner](retired/0039-qubit-params-json-owner.md)：typed handoff 由 0063 接替。
- [0040 — Autofluxdep run artifact](retired/0040-autofluxdep-run-result-artifact.md)：保存邊界由 0063 接替，workflow lifecycle 見 0062。
- [0041 — Autofluxdep feedback framework](retired/0041-autofluxdep-feedback-framework.md)：feedback 跨模組邊界接入 0062，slot 細節移至 app README。
- [0042 — Autofluxdep feedback confidence reversion](retired/0042-autofluxdep-feedback-confidence-reversion.md)：freshness 邊界接入 0062，公式移至 app README。
- [0043 — Autofluxdep runtime cfg override plan](retired/0043-autofluxdep-runtime-cfg-override-plan.md)：cfg 邊界接入 0062，snapshot／decoration 細節移至 app README。
- [0044 — GUI process runtime](retired/0044-gui-process-runtime.md)：跨模組啟動邊界由 0064 接替，局部契約見 gui README。
- [0045 — Shared GUI cfg core](retired/0045-shared-gui-cfg-core-ownership.md)：已被 [[0065]]、[[0073]] 的 Form owner 分工取代。
- [0046 — Shared cfg lowering ports](retired/0046-shared-cfg-lowering-ports.md)：已被 [[0065]]、[[0073]] 的 Form 轉換與持久 Spec 提交取代。
- [0047 — Expected errors](retired/0047-typed-expected-error-taxonomy.md)：跨模組邊界由 [[0068]] 接替。
- [0048 — Domain event facts and View reactions](retired/0048-domain-event-facts-and-view-reactions.md)：事件與反應由 0067 接替，pane matrix 見 measure README。
- [0049 — Lazy push](retired/0049-subscriber-aware-lazy-push.md)：跨模組邊界由 [[0068]] 接替。
- [0050 — Canonical cfg binding paths](retired/0050-canonical-cfg-binding-paths.md)：已被 [[0065]] 的 Form model-field paths 與 batch 契約取代。
- [0051 — Program cfg shape catalog](retired/0051-canonical-program-cfg-shape-catalog.md)：已被 [[0065]]、[[0073]] 的持久 Spec 與生成器取代。
- [0052 — Event attribution](retired/0052-event-meta-and-frontend-attribution.md)：跨模組邊界由 [[0068]] 接替。
- [0054 — Resonance multiplicative amplitude background](retired/0054-resonance-multiplicative-amplitude-background.md)：real log-amplitude slope 乘完整 resonator response，`edelay` 維持唯一 global phase slope。
- [0055 — Route-scoped resonator electrical delay](retired/0055-route-scoped-resonator-electrical-delay.md)：boundary search 可 bounded adaptive expansion；成功 delay 與 generator/readout route 以單一 compound calibration 持久化為後續 branch seed。
- [0056 — Resonance phase curvature and rational initializer](retired/0056-resonance-phase-curvature-rational-initializer.md)：optional resonance-centered quadratic phase background 與 internal degree-1 rational initializer 不接管 route-scoped delay branch。
- [0057 — RecordTable v2](retired/0057-flat-sampletable-v2-coordinate-contract.md)：座標保存契約由 0063 接替，schema 細節由 record_table owner 維護。
- [0058 — Registry-owned disconnect](retired/0058-registry-owned-visa-session-disconnect.md)：跨 owner teardown 由 0066 承接，registry 局部契約見 device README。
- [0059 — MCP RPC catalog](retired/0059-measure-mcp-rpc-channel.md)：跨模組邊界由 [[0068]] 接替。
- [0060 — Measure agent interface](retired/0060-measure-agent-interface-shared-gui-view.md)：跨模組邊界由 [[0068]] 接替。
- [0061 — Measure interactive plugin session](retired/0061-measure-interactive-plugin-session.md)：session ownership 由 0067 接替，互動細節見 measure README。
- [0077 — Numerical candidates、plan ownership 與 checked replay](retired/0077-numerical-plans-and-replay.md)：Numerical planner 與 checked replay 已移除，現行分工見 [[0078]]。
- [Domain guard 的 typed Permit proof](retired/typed-permit-proof.md#現況)：摘自 ADR-0001，領域檢查與 RunInputs 見 [ADR-0066](0066-operation-lifecycle.md#問題與決策)。
- [Operation handle 舊模型](retired/operation-handle.md#2-operation-handle歷史設計)：摘自 ADR-0002，現行 runner 與 lease 分工見 [ADR-0066](0066-operation-lifecycle.md#問題與決策)。
- [Cfg 編輯模型的局部取代範圍](retired/cfg-editing-replacement-scope.md#局部取代範圍)：摘自 ADR-0065，欄位轉換與草稿還原見 [ADR-0079](0079-form-field-transitions.md#決策狀態與問題)。
- [Measure 表單推導的局部取代範圍](retired/measure-form-derivation-replacement-scope.md#局部取代範圍)：摘自 ADR-0073，欄位轉換與草稿還原見 [ADR-0079](0079-form-field-transitions.md#決策狀態與問題)。
- [Pydantic 表單、輸入與宿主提交](retired/pydantic-form-tool.md)：superseded，穩定邊界併入 0065／0062，持久 Spec 與推導由 0008／0073 承接；原稿留存歷史。
- [量測成果 Artifact、設定快照與讀回](retired/artifact-persistence.md)：superseded，由 0063、0069、0067 接替；保留原提案與討論連結供追溯。
- [Cfg 編輯接縫與使用邊界](retired/cfg-editing-boundaries.md)：superseded，由 0065、0008、0073 接替，正文只供歷史追溯。
- [Cfg 資源公開契約與同步發布](retired/cfg-resource-contract.md)：superseded，由 0065、0008、0073 接替，正文只供歷史追溯。
- [具名 workflows 與可丟棄的步驟執行](retired/workflows-engine.md)：retired，保留期 0 候選草稿，workflow 執行契約由 0062 接替。
- [Autofluxdep 逐項宣告依賴與 predictor 載入](retired/autofluxdep-explicit-dependencies.md)：autofluxdep 已移除，不再提供 orchestrator 與 predictor 載入流程。
