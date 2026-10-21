---
date: 2026-10-13
status: superseded
superseded_by: docs/adr/draft/simulation-environments.md
related:
  - docs/adr/draft/simulation-environments.md
  - docs/adr/0064-process-startup.md
  - docs/adr/0069-experiment-core-and-frontends.md
  - docs/adr/0072-module-boundaries.md
basis:
  - "repo@007a30ccc5e25d4d68300b5c6ea2ddb9dfa693b5，撰寫本快照時"
  - "repo@179882fd00d583bffabe86c6b99ca54c796a92dd，第一輪速度 prototype"
  - "repo@ff31a20528fa8dc108fdbb93192fe98696905d5a，多體投影與 conditional Ramsey prototype"
  - "repo@6c6e164e69ea4d426a123211ed2dc00122226768，dispersive readout prototype"
  - "本機 prototype 環境：Python 3.13.11、scqubits 4.3.1、QuTiP 5.3.0、NumPy 2.4.6、SciPy 1.17.1、Matplotlib 3.11.0"
---

# 模擬量測環境、低能投影與 FTF preset 的討論

## 結論

截至本次討論，離線 prototype 已能在聯合 FTF 模型上產生可校準的 conditional Ramsey 資料，並經由帶雜訊的 I/Q 找回條件相位 π。這支持繼續設計可使用的模擬環境，沒有必要先把完整密度矩陣誤差追到很小。

工作重點因此從數值 benchmark 轉向 FTF preset 的參數表、第一版物理範圍及 acquire interface。使用者提出的 Python environment 配合 preset YAML，讓一般調參不必修改模擬程式。

已確認的設計方向由 [模擬環境 ADR draft](../adr/draft/simulation-environments.md) 擁有。本篇保存當時的比較與推導，不是完整規格，也不宣告正式功能已實作。

## 問題

目標是讓 agent 在接入真實量測前，經既有 GUI／MCP 練習單 qubit、多 qubit、coupler 與 FTF two-qubit gate。FTF 指 fluxonium–transmon–fluxonium。練習需要對 pulse、裝置設定與讀出條件作出有物理意義的反應，不能只回傳外觀合理的曲線。

當時有三個主要疑問：多體含時模擬是否太慢、讀出是否需要把 resonator photon states 一起演化，以及使用者如何建立和調整自己的環境。第一個 gate 目標是 CZ 類量測，但討論沒有先指定一種硬體 pulse 機制。

使用者也把 agent 評分、成功判準與知識接受制度排除於這項基礎設施設計。模擬來源可以追溯、真值可以事後對照，不等於本次要建立訓練或評量系統。

## 考慮過的選項與比較

### 固定 SimParams，或可組合的 Python environment

討論一開始就偏向簡單的 Python 宣告，而不是新造通用電路語言。使用者說：「我第一版想做簡單一點的，不要做出什麼複雜的DSL架構，可以考慮宣告本身是python腳本」。

後續又釐清 environment 不只是填入 SimParams。它可以選擇及組合自己的模擬實作，同時重用 eigensystem、propagator 等模擬工具。GUI／MCP 的生命週期仍由框架處理，不需要每個環境重新實作。

現行啟動與 coordinator 的事實分別見 [啟動 ADR](../adr/0064-process-startup.md) 與 [裝置環境 ADR](../adr/0069-experiment-core-and-frontends.md#顯式-cfg-與裝置環境)。不能把本次尚未實作的啟動固定模式與隔離目標，寫成既有功能已保證的行為。

### 持續的物理歷史，或 acquire 時捕捉輸入

曾考慮跨量測保存狀態、背景時間及漂移。使用者選擇較簡單的語意：「每次program acquire當下記錄模擬需要的參數，然後生成數據回傳。跨acquire無狀態。」

這不表示裝置值也要每次重設。Bias 等目前設定仍由環境持有；不延續的是模擬執行端的跨 acquire 量子態與演化歷史。純數值快取也與物理狀態不同，前者可以保留，不能重用已抽出的 shots 或 noise。

當時並未因此決定單次 acquire 內的初始化、reps、rounds 或 reset。先前提出過的具體量子態生命週期，不能反過來當作已獲使用者採納。

### 完整乘積空間、RWA，或全域低能投影

使用者要求平均每個掃描點約一秒，並授權 prototype。前幾輪測試確認簡單案例可以很快，但單 qubit 不能回答多體成本。使用者直接指出：「單qubit不管哪種算法都能很快，你的重點應該是多qubit系統」。

多體測試保留了真正的聯合 Hamiltonian、兩路不同 carrier、coupler 控制與耗散。直接在完整乘積空間積分的成本偏高。部分 RWA 與粗分段解法雖然快，卻未保留足夠準確的量測或 state 行為，因此沒有因速度好就直接採用。

使用者提出：「你也不一定要完整的4x3x4維度，你可以取整個系統在沒有外部驅動下的前M個能階就好吧？」這把兩個不同的維度拆開了：建立 H0 的基底可以較大，演化只需保留有用的 dressed states。控制、初態與耗散須一致投影，不能只截 Hamiltonian 的對角元素。

75／144 維 parent basis 投影到 M=8 後，完整 carrier 小掃描平均約 0.45 至 0.47 秒。名義點最大聯合機率差異約 0.0021。這是可行性證據，不足以把 M=8 固定成所有 pulse 的保留空間；中間 M 的 state 誤差也不保證單調。

### 高精度 gate 物理，或能運作的 gate 量測

使用者修正精度目標：「準確度要求不會太高，只要two qubit gate測量能運作就好」。這使驗證重點轉為實際量測行為，而不是繼續加大 M、追逐完整 state 誤差。

之後的 conditional Ramsey prototype 從帶 shot noise 的 fringe 找工作點，再用未參與搜尋的資料核對。它另外設定了示範耦合與 coupler bias，沒有假稱原先速度案例自動適合做 CZ。準備和 analyzer 用理想局部旋轉；gate 本身仍由耦合 Hamiltonian 演化，不直接套入預設 CZ 矩陣。

Hold 約 426.26 ns、含 ramp 總長約 626.26 ns 時，新資料量到 1.00445π。關掉耦合後相位差接近零，連做兩次 gate 則回到接近 2π。這支持量測流程可運作，不是 gate fidelity 或完整 CZ 校準的證明。

### Resonator 量子演化，或 dispersive readout response

使用者提出：「含時模擬不將resonator的維度納入，但可以模擬dispersive shift，並推算S12曲線，要考慮readout gain/freq/length等，也要考慮noise與平均次數」。

這個方案把量子演化與經典讀出響應分開。Prototype 從 FTF 的虛躍遷估算 frequency pull，再用兩個 resonator 共用傳輸線的複數響應處理有限 pulse、ring-up、demodulation 與積分。當時的線路採 co-located、reciprocal 近似；S12 與 S21 相等不是對任意線路的承諾。

Single-shot 先抽聯合狀態，再加接收雜訊。平均模式保留狀態抽樣及接收雜訊兩部分。分析端只用已知準備的校準 I/Q 與新量測的 I/Q，不讀隱藏 population。

I/Q-only conditional Ramsey 找到 hold 約 423.39 ns，新資料確認 0.99805π。平均次數從 16 增至 1024，I/Q 波動縮小約 7.95 倍，接近獨立抽樣預期的八倍。Gain 降低時 cloud 更難區分；gain=0 時校準被拒絕。

這些結果是在弱讀出、線性 resonator、讀出期間狀態固定的近似下取得。沒有模擬強讀出 transitions、T1 跳變、非 QND backaction 或 resonator Kerr。

### 三個不同 workload 的成本

| Prototype | 平均計算成本 | 不能混為一談的範圍 |
| --- | ---: | --- |
| M=8 聯合 FTF 完整 carrier | 約 0.45 至 0.47 秒／點 | 兩路 microwave carrier 加 coupler 控制，尚未經完整讀出鏈 |
| Conditional Ramsey | 約 0.061 秒／冷點 | Coupler detuning gate，局部 pulse 視為理想操作 |
| Gate 加 dispersive I/Q | 約 0.066 秒／冷點 | 重建模型並生成 4,096 shots，局部 pulse 仍為理想操作 |

測試使用 Intel Core Ultra 7 258V，BLAS／OpenMP 各一個 thread。數字不含 imports、GUI、program 編譯及檔案保存。三個 workload 不同，不能把表中的差距當作同一算法的加速比，也不能當作所有 acquire 的 latency 保證。

### FTF preset 與 YAML

在讀出 prototype 完成後，使用者提出：「我覺得可以有FTF的enviroment或preset，內部有個yaml檔來放一些超參數，這樣方便用戶調」。

這個提議並沒有要求把所有 Python environment 改成 YAML。它解決的是常見環境的調參成本：Python 保留組裝能力，FTF preset 接受一份有明確欄位與單位的參數檔。拓樸或新模型仍可用 Python 改寫。

討論也把「超參數」拆成物理真值、耗散及讀出參數、數值截斷、裝置初始值。它們與當次 program 的 gain、frequency、length 不同。若把後者也藏在 preset 裡，就可能讓使用者掃描的值被另一份設定蓋掉。

具體分工由 [ADR draft](../adr/draft/simulation-environments.md#模型參數裝置設定與量測輸入分開) 記錄。當時尚未凍結 YAML 欄位、預設值或 `build_environment()` 的具名回傳型別。

## 被否決或未採納的選項與理由

- 新造通用電路 DSL、preset registry 或自動探索。使用者要的是容易調整的環境，不是另一套框架。
- 為了簡單而讓所有 environment 只能填固定 SimParams。這無法涵蓋自訂電路及不同模擬實作。
- 只用單 qubit 的快速度推論多 qubit 可行。使用者要求真正聯合 FTF 的證據。
- 把某次 M=8 或 RWA 的成功當作普遍精度保證。截斷適用性與 pulse 有關，部分近似已有失準證據。
- 先把 gate fidelity 做得很高，再接量測流程。這不符合使用者後來明確降低的精度需求。
- 用兩個固定 I/Q 點或只在平均 population 上加 noise，當作完整 single-shot 模型。這無法同時解釋頻率、長度、聯合抽樣與平均次數的變化。
- 讓未支援操作默默回退白雜訊、正常硬體模式或舊值。這會把設定失敗偽裝成量測結果。

## 留下的問題

- 第一版 coupler 的控制語意。Prototype 的 number detuning 不是完整的 flux-dependent circuit，不能直接升格成正式 flux 模型。
- Shot 初始化與同次 acquire 內的狀態生命週期。跨 acquire 無狀態沒有替這題作答。
- FTF preset 的欄位、單位、預設值、偏壓映射及合法範圍。
- 量子演化、讀出與 program acquire 的正式輸入／輸出型別，以及多通道結果的 shape。
- 讀出模型第一版的能力限制、來源與 seed 紀錄，以及盲測時如何限制真值存取。

後續應先收斂這些設計，再做最小的實際接線驗證。沒有新的具體不確定性時，不再擴大數值 prototype。

## 依據

- 已確認方向：[模擬環境 ADR draft](../adr/draft/simulation-environments.md)。現況分工見其引用的啟動、裝置環境及模組定位 ADR。
- 當時的工作紀錄（decision map、使用者引文、逐輪 evidence）、多 qubit dressed、conditional Ramsey 與讀出的數值 prototype，以及其程式、raw I/Q、S12、JSON 結果與 SHA256 manifest，都是一次性工作檔，未入庫。本快照保存其結論與限制，不代替原始產物。
- Frontmatter 的 repo commits 是各輪當時的 tracked tree 基準，不包含 gitignored prototype；軟體版本以各輪 manifest 為準。

## 後記（2026-10-14）

以上保留2026-10-13的討論快照。後續已確認固定座標 coupler／固定投影、獨立 Gibbs shots、多能階 relaxation 與條件平均讀出，以及固定 ADC 訊號契約。輸入改為 acquire 固定量測要求、逐 round 捕捉環境；第一版僅支援同步模擬 acquire，不採分步使用權或 abort 設計。現行理由與限制由[模擬環境 ADR](../adr/draft/simulation-environments.md)擁有。

完整 YAML schema／預設值、具體 Python 型別、保存與遷移細節及完整效能仍未完成，不能將前文「留下的問題」全部視為已結案。T1／pointwise Q／Purcell 的單位修正另已交付，契約見 [ADR-0071](../adr/0071-coherence-units.md)。原型不是完整 FTF 或每點效能的驗收證據。

當時的規格交接與研究狀態是未入庫的工作文件；現行決策請讀 ADR。
