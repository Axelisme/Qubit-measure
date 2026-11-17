---
date: 2026-11-17
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/draft/simulation-environments.md
  - ../../lib/zcu_tools/simulate/adc/README.md
  - ../../lib/zcu_tools/simulators/single_fluxonium/environment/README.md
  - ../../lib/zcu_tools/simulators/six_fluxonium/environment/README.md
basis:
  repo_commit: ed552a79b287db464bce43c1414252f221f16fdb
  acceptance_commit: 60cc37798469a6e7aee259f7c2f9958bde5cc606
  external_sources: none
---

# Projective read 的輸出改成 Markov chain

## 結論

Single／six 有 projective read 時，single runner 不再列舉 outcome history，改輸出 `simulate.adc.ProjectionChain`：每點第一個 read 的 populations，加上每個後續 read 依前一個投影狀態的轉移表。ADC 的 raw／trace 逐 shot 沿 chain 抽狀態；integrated 用逐態 multinomial flow counts，各 event 的 histogram 以累積邊界編成共同 bins。編碼器歸 `simulate.adc`，six 的 integrated 合併改呼叫它。2 GiB 預算與它的錯誤移除。FTF 不動。

`projection_prune_floor` 的欄位、範圍與預設不變，語意改成：在某一點近似機率低於 floor 的投影狀態，在該點不續演化，近似為留在原狀態。判斷逐點進行，一點的近似不依同次 acquire 的其他點。

這是 designer 依使用者授權的代裁（2026-11-17），寫作時使用者尚未覆核。現行規則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#projective-read-的-markov-chain-表示) 與[模擬環境草稿](../adr/draft/simulation-environments.md)。

## 問題

使用者 2026-11-16：「模擬器不該有 OOM」，看過選項後說「OOM你就改designer決定」。預設 `chain/amp_rabi` 是 51 個 gain 點、levels=21、initial delay 150 µs、15 次 projective read 加 terminal，共 16 個 read event。它在 single fluxonium 模擬器上被 projection 的 2 GiB 預算拒絕。題目要選出最小、能讓它在 8 GiB process cap 內完成的設計。

成因（數字取自調查，原始工作檔未入庫）：

- Projection runner 把每條 outcome history 攤成 dense 輸出：`weights (P,K)` 與每個 event 一份 `populations (P,K,S)`。K 隨 event 數指數成長；單點 levels=21 時 K=183,665（實測），populations 471 MiB。51 點時一份輸出估算 23.5 GiB，建構複製後 47 GiB（算術，固定 K 假設）。
- ADC integrated 又展開成 `(outcome, terminal state)` 共 K·S 個 category。每個 channel 的 conditional means 單點約 0.92 GiB，51 點約 47 GiB（算術）。
- 51 點在第 12 次 split 被擋：候選 K·S＝1,089,501，workspace 約 2.48 GiB（實測）。單點 levels=21 跑完要 15.5 s、峰值 RSS 3.94 GiB；levels=4–21 的 51 點全部被拒絕（實測）。
- 投影是 DC rank-1，每段 span 的輸入只有 plan、point、span 時間與投影後的 density。所以 floor=0 時完整路徑機率是 `initial[x0]·Π T_e[x_{e-1}, x_e]`，資料本身就是 Markov chain。
- 下游只用總和：program 求平均只讀 `ChannelStatistics.sums`；six 合併器已自己用累積邊界產生人工 bins。

第二個問題在實作時出現：「機率在至少一點 ≥ floor」的「點」取哪個範圍。一個 request 可以有多個 plan。Native 前端每次 acquire 只產生 1 個 template（`chain/amp_rabi` 51 點、`twotone/chain` 301 點各 1 個）；多 plan 只出現在 six，同一 route pattern、同一 cavity 內 active incident tone 數不同的點分成不同 template。

## 考慮過的選項與比較

### 輸出表示

| 選項 | 做法 | 結果 |
|---|---|---|
| 維持 dense 加 2 GiB 預算 | 只改文件，寫明可行範圍 | levels=21 最多 2 點；51 點在 levels 4–21 全被拒絕 |
| Caller 每次 1–2 點 | acquire 把點軸切小批，每批用完丟掉 readout | 約 13 分鐘（15.5 s×51，未實測）；deferred raw 要保留 dense 輸出或改延遲抽樣契約；每批仍有 K·S 展開 |
| 只給 integrated 充分統計量 | integrated 用每個 read 的邊際計數 | raw／trace 仍要 dense；丟掉 read 之間的路徑相關 |
| 緊湊 history | int history 加依末態的 terminal | K 仍指數成長，第 12 次 split 已 2.48 GiB |
| **Markov chain（採用）** | 每點初始分佈加每個後續 read 一張轉移表；integrated 用逐態 flow counts | 從零設計也會選這個表示；chain 51 點約 2.7 MB，integrated bins 每 channel 約 4.2 MB，都不隨 reps 成長 |

Integrated 的 flow counts：`N0 ~ Mult(reps, initial[p])`；每個後續 event，對每個來源狀態 s 抽 `F[s,:] ~ Mult(N_prev[s], T_e[p,s,:])`，`N_next = Σ_s F[s,:]`。各 event 的 state histogram 用累積邊界編成一條共同 bins 軸，每點最多 `1+E(S−1)` 個，本例 321。各 read 的 noiseless totals 的聯合分佈與逐 shot 抽完整路徑相同（以推導與 `Fraction` 窮舉核過）。

### floor 的語意

採用 identity 列：低於 floor 的狀態不續演化，該列設 identity（留在原狀態）。不丟機率、不重新正規化、沒有空列。誤差界：固定同一組 solver 轉移列與 initial，路徑分佈與精確 chain 的 total variation ≤ Σ_t Σ_{被設 identity 的 s} q_t(s)，q_t 是該點自己的近似 source marginal。floor<1/S 時每個轉移至多 S−1 個狀態低於 floor，上界是 15×20×floor＝3.0e-7（floor=1e-9）。這個界不含 solver／routing 誤差。單一 read 的 expected I/Q 誤差除以 I/Q 中心直徑 ≤ TV；條件化在機率 π_A 的 herald 事件上時界放大成 2TV/π_A，π_A ≥ 3.0e-4 才保證 ≤2e-3。對照：原 history 剪枝在 levels=21 單點丟掉 1.19e-4 的機率（實測）。

### Active state 的範圍

| 範圍 | 規則 | 對外說明 |
|---|---|---|
| **逐點（採用）** | 點 p 的近似機率 `q_t(p,s) < floor` 時，p 的第 s 列設 identity；每個 plan 對「plan 內至少一點需要」的狀態聯集演化一次，某點不需要的列算過也不用 | 「在某一點機率低於 floor 的投影狀態，在該點不續演化」，不提 plan／template |
| 逐 plan | 狀態在 plan 內每點都低於 floor 才整個 plan 設 identity | 要寫「同一結構 template 的所有點」 |
| 跨 request | 跨所有 plans 取聯集，迴圈改 event 外層、plan 內層 | 要寫「同一次 acquire、同一 cavity、同一 route pattern 的所有點」 |

三者在呼叫數與精度上沒有誰普遍勝出。以 3 狀態、2 點、2 轉移的有效 stochastic kernel 用 `Fraction` 窮舉，找到逐點呼叫較少（5 對 6）與逐點 TV 較小（0.12·floor 對 0.98·floor）的例子，也找到跨 request 比逐 plan 少呼叫的例子（10 對 11）。不同範圍第一次用了不同的列之後，後續的近似 marginal 就分岔。三者的誤差上界相同。

剩下的差別是「一點的近似是否依同一次 acquire 的其他點」。沒有使用者裁定或文件回答這件事。逐點在兩種答案下都成立，也不比另外兩者貴，所以採用逐點，這個缺口不擋裁決。

Diagnostics：`continued_states[p][t]` 是點 p 在轉移 t 採用 solver 列的狀態數，以採用 mask 計數，不以「矩陣不等於 identity」判斷；`uncontinued_probability[p][t]` 是同一 mask 補集上 `q_t(p,s)` 之和；t 共 reads−1 個。`evolution_calls` 記含 prefix 的實際呼叫。有「算過但本點沒用」的列，所以 calls 不能由採用數換算。

### 先例與立場

| 先例 | 強度 | 立場 |
|---|---|---|
| 使用者 2026-11-12「整理不要求輸出和整理前相同；要求誤差在標準內」 | 有拘束力 | 遵循：路徑分佈在誤差界內，抽樣值不同 |
| Light change `oom-projection-collapse`（main `ed552a79b`），使用者「採用建議」：依最後狀態共用 density，加 2 GiB 預算與 `ValueError` | 有拘束力 | 共用 density 遵循並延伸到輸出。預算區別：使用者要的是明確錯誤而不是 OS kill；記憶體改成隨點數線性後，預算沒有要擋的成長 |
| ADR-0078「outcome weights、各 event populations 與 conditional sums 是可重用的數值產品」 | 有拘束力 | 區別：規則不變，列舉的產品加上 chain 的初始分佈與轉移表 |
| ADR-0078「直接平均維持 conditional moments 和既有 RNG stream」 | 有拘束力 | 區別：那句記的是單 carrier CTMC raw 那次改動的範圍；有 projective read 的 single／six 改抽法，列為覆核項 |
| ADC README 的單 event RNG 順序承諾與 unprojected parity 測試 | 有拘束力 | 遵循：無 projective read 與 FTF 不經新路徑 |
| 原則〈模擬器不替使用者擋實驗參數〉 | 有拘束力 | 遵循：51 點、levels=21 是合法參數，預算錯誤等於替使用者擋 |
| ADR draft 的 floor 句（剪全域 history 權重並重新正規化） | 已核准方向 | 區別並改寫：欄位、範圍、預設不變，single／six 的語意改成逐點 identity |
| 選路歷史的使用者裁定：為速度可損失一致性，條件是每個可能 kernel 都在 2e-3 內且歷史可控 | 有拘束力 | 類比，遵循精度門檻；它沒有裁「其他點可否改變原點的近似」 |
| [Direct integration 討論](2026-10-18-direct-integration.md) 撤 gain grouping：結果不依 apply-call 組成 | 僅供參考 | 遵循：一點的近似不依同批其他 rows |
| 原 history 剪枝逐點判斷、plan 聯集演化 | 僅供參考 | 遵循 |
| Private replay 版本契約 | 有拘束力 | 遵循：single 升版，舊 archive 明確拒絕 |
| 記憶體上限要有 cap 下的峰值 RSS 證據 | 有拘束力（流程） | 遵循：驗收在 8 GiB scope 實量 |

## 被否決的選項與理由

- 維持現狀並說明可行範圍：預設 51 點在所有測過的 levels 都被拒絕。
- Caller 分批：約 13 分鐘；deferred raw 讓 dense 輸出仍要保留，或得改 raw 的延遲抽樣契約；每批仍有 K·S 展開。
- 只給 integrated 充分統計量：raw／trace 仍要 dense；丟掉 read 之間的路徑相關。
- 緊湊 history：K 仍指數成長。
- Integrated 用前綴樹抽本輪實際路徑當 category：category 數隨 reps 成長，10^6 reps 時單 channel 約 12 GiB。
- 低於 floor 的狀態丟掉並重新正規化：逐列與整體正規化給出不同的路徑律，還多出「某列全被剪光」的失敗。
- FTF 也改成 chain：FTF 的 history 帶腔記憶，DC 末態不足以決定後續。
- floor 退役、永遠精確：欄位保留不讀是相容邏輯；刪掉欄位會讓既有的 YAML（schema `extra="forbid"`）讀不進來；floor=0 已給精確結果。
- 逐 plan、跨 request 的 active state 範圍：不比逐點便宜或精確，對外說明要帶內部分組；跨 request 還要跨 plan 同步迴圈，「不隨分組改變」也只在一次 preflight 內成立。
- 跨 six 整個外部 acquire 取 active state：不同 cavity 沒有共用狀態軸，state index 無定義。

## 留下的問題

- 同一狀態在多個 plan 的 evolve 要不要併成一次 `Engine.evolve`：與逐點規則正交，native 只有 1 個 template 時沒有差別。併送會改變引擎看到的 batch 組成、reuse key 與峰值記憶體。等 six 多 template workload 的 profile 顯示 setup 佔比高時再開題。
- 記憶體隨點數線性成長（raw 另隨 reps），不保證任何 workload 都不會 OOM。
- Herald 機率 π_A < 3.0e-4 時，floor=1e-9 不保證條件化結果在 2e-3 內；floor 是精度旋鈕，設 0 給精確 chain。
- 驗收用 tracked 合成校準，不證明使用者本機校準選到同一個 kernel。
- 寫作時使用者尚未覆核的點：有 projective read 時同 seed 結果改變、舊 private replay archive 被拒；floor 意義改變且定為逐點；category 軸是人工 bins，不代表真實 shot 路徑；2 GiB 預算移除。

## 依據

- 調查與成本 probe 讀 main `ed552a79b`。Probe 用 tracked single fluxonium 預設模型（levels=21、floor=1e-9）、drive 580 MHz const 1 µs、gain −0.3..1.0、readout 7000 MHz gain 0.1 1 µs。原始工作檔未入庫。
- 51 點 projection 演化的成本 probe 用「目的狀態剪枝並重新正規化」，不是採用的 identity 規則：44.3 s、峰值 RSS 557 MiB、121 次 evolve（8 GiB scope）。Identity 規則估 121–316 次呼叫，316 次外推約 114 s。
- 驗收量測在 `60cc37798`，8 GiB／swap 0 scope。輸入是 tracked `tools.family_pareto_settings` 的 `SINGLE_MODEL`／`BOARD_PROFILE` 加上同一組 drive／readout 設定，不是使用者本機校準。前端由 `ChainAmpRabiExp` 編譯一次，核對 1 plan、51 點、16 read event、21 states；reps 100、floor 1e-9；不做 e2e。

| Levels | Seam | wall s | 峰值 RSS MiB | evolve 次數 |
|---|---|---:|---:|---:|
| 21 | producer | 70.0 | 778 | 139 |
| 21 | integrated | 0.11 | 406 | 0 |
| 21 | raw | 6.6 | 550 | 0 |
| 12 | producer | 9.1 | 480 | 144 |
| 12 | integrated | 0.06 | 399 | 0 |
| 12 | raw | 5.8 | 540 | 0 |

  RSS 含整個 payload 的 import 與 setup，wall 只包對應 seam；每組量一次，21 與 12 的差包含 JIT／cache，不能推論縮放倍率。逐點跨轉移 `uncontinued_probability` 總和的最大值：levels=21 為 1.875e-8，levels=12 為 1.864e-8。
- 有限 chain 的 `Fraction` 窮舉（3 狀態、2 點、2 轉移）是有效 stochastic kernel，不是物理 workload。
