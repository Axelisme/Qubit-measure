---
status: superseded
---

# 0077 — Numerical candidates、plan ownership 與 checked replay

**狀態：** superseded，2026-10-18。實驗 constant planner、shadow graph、profiles、calibration 與 checked plan replay 已移除，不提供相容入口。現行模組邊界與 dispatch／kernel 分工由 [[0078]] 接替，API 與決策圖見 [evolution README](../../../lib/zcu_tools/simulate/evolution/README.md)。以下正文保留歷史設計，不是現行契約。實驗結果見 [planner 討論快照](../../discussions/2026-10-18-planner-evidence-and-overhead.md)。

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
Kernel 擁有數值 guard、prepare keys、algorithm identity 與數值運算。
Planner 直接認得各算法，在各算法段落維護適用條件、成本特徵與大型配置估算。
新增算法時直接修改 planner，不建立通用 declaration、memory accounting 或成本 plugin 介面。
既有 declarations 保留為資料，不要求 kernels 回報全部 workspace。
Planner 不解讀或重造 kernel 的 opaque key。

2026-10-18，使用者確認這個耦合是刻意的設計。
原話為「看起來planner與使用的算法是本質高偶合，那planner可以清楚的區分每個算法是合理的架構，不需要特別抽象化，然後我認為planner先彙整全部資訊再決定每段的演化算法是正確的思路」。
Accuracy class 不是 family 的固定順位。

Caller 傳入數值 model、controls、accuracy policy 與 replay inputs。
Evolution 不讀資源容器或使用者套件，也不解讀 ADC 語意。
這個分工沿用 [[0072]] 的 simulate 定位。
沒有新增跨 package 的依賴方向，既有 C8、C9、C12、C16 繼續約束分層、無循環與 framework independence。
本篇不新增或放寬 import-linter contract。

### Experimental constant cover

`evolution.constant.ConstantPlanner` 是明確選用的 developer collaborator。
它直接為 Dyson、Hilbert period、T6、direct 與 T4 建候選，保留 declarations 的 identity 資料。
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
Search 先彙整完整 acquire，再比較完整 cover。
Work/build 下界可以省略不可能勝出的重複 graph scoring，但不刪除安全 incumbent 或改變相同成本的 tie-breaking。

Planner 只檢查 T6 的大型 d⁴ channel arrays 與其 allowance。
這不是 SciPy native workspace 的上界，也不保證 process RSS。
其他未知配置比照 legacy 不設新限制，`required_bytes=None` 保留總量未知。
完整已知的 synthetic memory charges 仍可用來驗證 shadow resource scoring。
Cache 的 opaque-byte limit 與 allocation/consumer lifetime 不變。

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

`OneCoreProfile` 單獨使用時仍是 uncalibrated，matching revisions 也只選 legacy。
`CalibratedConstantProfile` 包含這份 stage estimates、固定 numeric source closure digest、數值與 planner revisions、精確 workload/resource certificates，以及 cold/warm matched acquire evidence。
Source commit 記錄量測來源，不把整個 repository commit 當成 kernel stale gate。
固定 closure 的來源缺失或不匹配時不 admit。
完整 identities、數值依賴與 execution revisions 也必須匹配。

Production profile 只支援實際驗過的 source/model/input/control 與 worker/cache 容量。
Dimension 與 budget 符合，不代表整域適用。
每個 workload 的 cold/warm 整體 acquire wall time 都不得比 legacy 長，population 與 low-four density maxabs 都必須 <= 1e-2。
計時必須包含 admission、planning、guards、prepare、bind 與 apply。
Stage prediction errors 保留為診斷，不設獨立 admission 門檻。
目前沒有隨 package 發布通過這些條件的 profile。

`Engine.evaluate_constant_profile` 是明確選用的 developer 驗證入口。
它與正式 evolve 共用 planner、pool、cache 與 Schedule，只允許 matched acquire evidence 尚缺。
Stale revisions、域外輸入或資源、已有失敗 evidence 都明確 raise，不偷偷量測 legacy。
入口不計時、不修改 profile，也不把回傳結果標成 production-safe。
外部 harness 量測完整呼叫，取得 accuracy 與 matched wall evidence 後才能建立完整 profile。

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

`Engine.evolve(execution="auto", constant_profile=None)` 維持既有 legacy dispatch。
明確選 `legacy` 或 `legacy-plan` 時忽略 profile。
只有 auto 加明確選用的完整 profile 通過事前 admission，才使用 constant cost planner。
Nonadmission 在 guards 與數值執行前選 legacy，不捕捉 build/apply failure 後重試。
沒有 profile 的 checked replay 不改 assignment。
這個入口仍不支援 multi-interval production admission，也不改變預設數值政策。

不採用以 uncalibrated profile 切換預設路徑，因為 deterministic cost search 不能證明 approximation 適用或效能可信。
也不把 raw shadow graph 當 replay instruction，因為 shadow 可以保留 tentative guards。
Replay 保存已驗的 numerical assignment，scheduler 才擁有執行順序、prepare cache 與取消邊界。
