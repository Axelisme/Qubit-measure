---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0073-measure-form-derivation.md
  - 2026-11-21-form-board-limits.md
basis:
  repo_commit: e2761947161b031245b1bf29c085a8436d2828a8
  main_base: 6e44c21321f8099993e20346fce2ff49daac5123
  external_sources: none
---

# Gain 固定範圍：表單只驗純量，掃描留給 Run

## 結論

使用者本人最後選擇只保留純量 gain 的表單限制。runtime `PulseCfg.gain` 與 `PulseReadoutCfg.gain` 重用既有 `Gain`，閉區間為 [-1, 1]；Form 對超界純量標出原欄位。`MyProgramV2.add_pulse` 的 Run 時檢查仍保留。

34 個一般 gain 掃描欄位維持 `SweepCfg`，不新增掃描範圍型別或表單端點投影。表單可接受超界掃描；超界點在 acquisition 前被拒。Host 迴圈把每點當純量寫入 runtime cfg，於建構該點時拋 `ValidationError`；FPGA sweep 的 `QickParam` 則在編譯時由 `add_pulse` 拒絕。兩條路徑的報錯位置與診斷不同，host 掃描較早的合法點可能已完成。

Mux Amp Rabi 仍由 `MuxAmpRabiCfg.validate_scan` 依 `participants` 檢查 tone 範圍；非參與 tone 的超界草稿不擋，參與 tone 的錯誤不標在 Start／Stop。CZ 的 `scale × gain` 仍留在 Run。Mux validator 與 host 掃描的報錯位置都是 designer 依授權代裁，不是使用者逐項裁定，且尚未經使用者覆核。`reset_check` 的預設 gain 上限另以 `min(1.0, ...)` 修正，與手動輸入的範圍檢查分開，見[表單預設值討論](2026-11-21-form-board-limits.md)。

原本擴大到掃描表單的代裁方案及其五種端點特例，在落地前全部撤回；沒有保留相容分支或 legacy 檢查。

## 問題

起因是 `reset_check` 的掃描 Stop 可能超過 1。預設值瑕疵與操作員手動輸入超界是兩件事：前者修正 seed，後者才涉及檢查放置。

`Gain` 已用單一欄位 annotation 表達純量範圍，但 runtime pulse cfg 起初沒有套用它。一般 `SweepCfg` 也沒有 gain 語意；表單可用 Range、Center 或 Step 編輯掃描，而有效端點可能是推導值。Run 原有的 `add_pulse` 範圍檢查則保護 QICK 呼叫與非表單呼叫者。

## 考慮過的選項與比較

- **純量與掃描都在 Form 檢查：** 重用 `Gain` 限制純量，再加掃描 bounds metadata、`GainSweepCfg` 與 form adapter 的有效端點投影。方案能把掃描超界標在表單列，並集中維護範圍值；但需把檢查接到 34 個掃描欄位與四種輸入模式。
- **各實驗自行驗證掃描：** 不需通用投影，但同一範圍會重複宣告；core validator 的錯誤也不會自然成為 Form 的欄位紅字。
- **只限制純量：** 掃描維持一般 `SweepCfg`，超界點由 Run 路徑拒絕。這只增加兩個 runtime 欄位的既有 annotation，不另造掃描抽象；使用者最後選擇此方向。

使用者最初以為「Pydantic 里面 annotation 可以给 G 一跟 L 一之类的」就能簡單限制欄位（2026-11-21）。這對純量成立；掃描還要將輸入模式投影成有效端點，不能只把同一個 annotation 套到 `SweepCfg`。

## 被否決的選項與理由

保留掃描表單檢查會讓簡單的單欄位限制長成一套需長期維護的投影機制。designer 代裁的方案後續累積了五種特例，使用者尚未逐項裁定或覆核：

1. Center／Step 沒有可標示的 Start 或 Stop 列，必須按有效端點與可修改輸入決定紅字落在 Center、Span 或 Step。
2. 輸入 Stop 恰為 1 時，`−0.7 + 0.1 × 17 = 1.0000000000000002`；嚴格比較會把浮點投影誤差當成超界，於是又加相對容差與端點夾值。
3. 夾值後有些輸入會違反原生 `SweepCfg` 的 Step 一致性，還要在特定情況重算交付的 Step。
4. 表單預覽與交付值必須共用夾值後的投影，否則畫面會顯示超界、實際交付卻已被夾到邊界。
5. 多點掃描夾值後可能兩端重合；四種模式都要再加整欄錯誤，避免同一個點被重複掃描。

這些分支分別增加錯誤列規則、容差與正規化、Step 協調、預覽同步和退化掃描拒絕。它們沒有改變正常量測行為，維護成本不符合使用者最後採用的判準。

逐實驗 validator 也沒有保留：範圍會散落在多個 owner，且不能自然提供所需的欄位紅字。Mux 是條件不同的既有例外：它依 `participants` 決定哪些 tone 要檢查；把它改成無條件 `Gain` 會連未參與 tone 的草稿一起拒絕。保留原 validator 是 designer 代裁，尚未由使用者覆核。

## 留下的問題

目前沒有第二種 sweep 類型提出同類的共用範圍需求。若將來出現，應先確認多種 sweep 都需要它，再評估是否值得新增通用型別與投影。

Host 迴圈與 FPGA sweep 的錯誤階段和訊息不同。這是 designer 為避免額外分流機制而代裁的行為，尚未經使用者覆核；若要統一診斷，需另行裁定，也不能把先前已完成的合法點視為回滾。

## 依據

使用者對簡單固定範圍檢查的原話是「2檢查固定範圍這種簡單的機制可以在form做，太複雜的保留在run抱錯」（2026-11-21），並核准以此作為表單與 Run 的分界。

維護原則由使用者的原話整理而來：「能够简单实做的机制，那就可以实做。如果需要加上一些需要维护的机制，那就算了。检查这种东西简单就好，对我来说，维护的负担才是重点，因为这个库呢，这个 repo 只有我一个。人类维护者，可能再加上一些 agent，我不可能花心思去维护每个细节，所以好维护对我来说很重要。目前行数已经六十万行了，我不可能看完了，agent 一个 session 也不可能读得完，所以呢，纪律大纲。还有维护性是重点。」（2026-11-21）。使用者接著確認：「例子没错，边界也对。简单来说，不影响行为的检查应该以自然、简单即可。」（2026-11-21）。

2026-11-11，使用者曾說「特化一个 gain sweep config，感觉就有点过度工程」，並以多種 sweep 是否都有檢查需求作為建立通用型別的條件。這句來自未入庫的對話紀錄；原始紀錄未入庫。早期掃描擴充的代裁漏查了這個先例。

使用者最後表示「那就采用建议吧」，並以維護負擔作為理由（2026-11-21）。其裁定是只留下可直接重用的純量 `Gain`；掃描投影與上述特例是 designer 先前代裁、後來撤回的方案，不應寫成使用者逐項決定。

程式與決策依據：[`ADR-0073`](../adr/0073-measure-form-derivation.md)；[相關表單預設討論](2026-11-21-form-board-limits.md)；[`Gain`](../../lib/zcu_tools/cfg_model/gain.py)、[`PulseCfg`](../../lib/zcu_tools/program/v2/modules/pulse.py)、[`PulseReadoutCfg`](../../lib/zcu_tools/program/v2/modules/readout.py) 及 [program/v2 modules README](../../lib/zcu_tools/program/v2/modules/README.md)；[program/v2 README](../../lib/zcu_tools/program/v2/README.md) 與 [`MyProgramV2.add_pulse`](../../lib/zcu_tools/program/v2/base.py)；[`MuxAmpRabiCfg`](../../zcu_lab/v2/mux/amp_rabi/core.py)、[`CZAmplitudeCfg`](../../zcu_lab/v2/two_qubit/cz_amplitude/core.py) 及 [`reset_check` form](../../zcu_lab/v2/singleshot/reset_check/form.py)。
