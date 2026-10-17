# 0076 — Numerical candidates、plan ownership 與 checked replay

## 脈絡

Legacy dispatch 先按條件選 family，再由 scheduler 共用 preparation 與執行 timeline。
這個順序不能比較共享 build 與整個 acquire 的執行成本。
但把未校準成本接進預設 dispatch，會讓成本模型的錯誤直接改變 production physics。
兩者不能同時切換。

2026-10-17，使用者同意 span-aware planner 的選擇方向，並要求與既有數值優化平行。
原話為「B 1b 2同意 3同意 4可以 然後這是大重構，我建議與現有的優化規劃平行，也就是說一個或多個subagent負責重構，同時舊架構繼續推算法優化」。
本篇落實 candidate、plan、準確度與 replay 的責任。
Model reduction 與 exclusive CPU lease 的未實作邊界不在本篇宣告。

## 決策

### Numerical owner

`simulate.evolution` 擁有 numerical candidate、完整 cover、cost search 與執行 adapter。
Kernel 擁有 cheap eligibility、數值 guard、prepare keys、algorithm identity 與數值運算。
Planner 呼叫這些契約，不解讀或重造 kernel 的 opaque key。
Accuracy class 不是 family 的固定順位。

Caller 傳入數值 model、controls、accuracy policy 與 replay inputs。
Evolution 不讀資源容器或使用者套件，也不解讀 ADC 語意。
這個分工沿用 [[0072]] 的 simulate 定位。
沒有新增跨 package 的依賴方向，既有 C8、C9、C12、C16 繼續約束分層、無循環與 framework independence。
本篇不新增或放寬 import-linter contract。

### Experimental constant cover

`evolution.constant.ConstantPlanner` 是明確選用的 developer collaborator。
它以現有 Dyson、Hilbert period、T6、direct 與 T4 declarations 建候選。
目前只接受每條 timeline 一個 start=0、fixed-flux、零或一個 constant tone 的 interval。
Sampled controls、多 interval、continuous flux 與未知 prefix input 明確失敗，不刪掉不支援的 rows。

Planner 對每個 eligible option 執行真正的數值 guard，保存 row acceptance 與完整 source-input digest。
Cheap eligibility 或 cache proof 不能取代 acceptance。
Source density、DC 及 row models 的實際數值都進入 digest，不只比較 caller 的 identity 字串。
本切片支援現有 no-build constant guards。
注入需要 guard-build handoff 的 variant 明確失敗，不把 call-local guard tables 放進 replay record。

T4 與 T6 必須覆蓋其完整 no-tone 與 single-constant-tone domain。
缺少 terminal 或 terminal rejection 直接失敗。
Planner 保留這個已接受的完整 incumbent，讓其他已通過 guards 的方法競爭成本。
它呼叫既有 bounded shadow search，不另造 dispatch tree 或 scheduler。

`ShadowPlan.production_safe` 一律為 False。
Raw shadow graph 不能執行。
Constant adapter 只把有完整 coverage、匹配 input certificates 與完整 kernel identities 的 numerical cover 交給既有 `Schedule`。
Graph 中的 task order 仍是評分資料，不是任意可執行指令。

### Accuracy 與可重現性

Guard 與一核心 coefficients 不隨 workers 改變。
Logical workers 改變全 acquire graph 的容量，因此可以改選 family，也可以選 mixed cover。
相同 policy、profile、inventory、resource request 與 planner revision 產生相同 assignment。
固定 tie-breaking 與 search bounds 不依 elapsed time 或工作完成順序。

每個數值 guard 仍擁有原本的適用條件。
Segment budget 不是端到端誤差證明。
Synthetic costs 只能驗證搜尋與 tiny numerical contracts，不能決定真實 production 模擬的數值方法。

Guard rejection 排除尚未執行的候選。
Guard、build、bind 或 apply exception 直接傳遞，不捕捉後改選方法。
Post-apply exceedance 保留未修補輸出，單次 replay 只發一次彙總 warning。
Planner 不正規化 density，也不做事後重試。

### Profile 與 kernel revisions

Profile 記錄量測時的完整 algorithm identities，包含 revision、parameters 與 execution variant。
`OneCoreProfile.production_admission(current)` 在 assignment 前比對目前需要的 identities。
Revision 不符時回傳明確的 legacy decision 與原因。
這不是捕捉數值失敗後的 fallback。

目前所有 `OneCoreProfile` 都是 uncalibrated。
即使 revisions 相符，admission 仍選 legacy。
實際 production admission 還需要 matched guard/build/bind/apply calibration 與 held-out acquire evidence。
本切片不提供可繞過這些條件的 calibrated flag。

Calibration 由 planner 工作負責，量測時取得共用 profiling lock，使用當時 main 的 kernels。
Kernel 優化可以繼續，不必為 profile 凍結。
舊 profile 在 kernel revision 改變後失去 admission 資格，重新校準後才可再申請。
另一個工作的整體 profiling 報告不能取代這份 matched evidence。

### Pinned replay

`PinnedConstantPlan` 保存選定 numerical cover、完整 segmented controls、candidate parameters、
profile ID/revision、constant 與 shadow planner revisions、numeric revision、kernel identities、
row/input certificates，以及原 resource/cache assumptions digest。
Record 不保存 evolved density 或 prepared table。
Caller 仍提供匹配的 numeric source 與 batches，這不是新的使用者存檔格式。

Replay 先驗完整輸入與 revisions，再把固定 assignment 交給原 scheduler。
Cache eviction 或不同的 cache capacity 可以重建同一 preparations，不能觸發 assignment search。
不支援的 planner、numeric 或 selected kernel revision 明確失敗。
這和 profile revision mismatch 時的 production non-admission 是兩種不同結果。

目前 replay 只 inline 執行，沒有 worker-width 參數。
Logical widths 1、2、6 驗證 planner capacity，不代表已支援跨實際 worker width 的 replay。
日後 executor 可以在不同容量排程同一 assignment，但不能因此改選 physics。

## 後果與替代方案

預設 `Engine.evolve` 與明確選用的 `execution="legacy-plan"` 都不呼叫 constant cost planner。
開發者可以先驗 numerical ownership 與 replay，production 仍使用既有 dispatch。
代價是這個切片不處理多 interval acquire，也不交付校準或效能改善承諾。

不採用以 uncalibrated profile 切換預設路徑，因為 deterministic cost search 不能證明 approximation 適用或效能可信。
也不把 raw shadow graph 當 replay instruction，因為 shadow 可以保留 tentative guards。
Replay 保存已驗的 numerical assignment，scheduler 才擁有執行順序、prepare cache 與取消邊界。
