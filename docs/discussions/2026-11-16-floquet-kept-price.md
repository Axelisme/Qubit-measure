---
date: 2026-11-16
status: active
superseded_by: null
related:
  - docs/adr/0078-evolution-module-boundary.md
  - lib/zcu_tools/simulate/evolution/README.md
  - docs/discussions/2026-10-18-two-tone-floquet.md
  - docs/discussions/2026-10-18-planner-evidence-and-overhead.md
  - docs/discussions/2026-11-16-floquet-refusal-cost.md
basis:
  repo_commit: f33f7b51ff5d52326eb6db3c45f6cc4e44694f96
  external_sources: none
---

# Floquet 的保留與價格重校

## 結論

當時在 committed 批次上，`floquet` 比 `dop853` 快 1.65–1.81 倍，因此保留並重校價格。
這是下文「保留方案配對量測」的結果，不是所有輸入上的速度保證。
三個 family 的價格排序等 `multitone_rwa` 修好後再做。
這次只校準 `floquet`／`dop853` 的價格，沒有更改 guard、放行門檻或候選順序。

使用者於 2026-11-14 選擇再研究一輪，並設停損條件。
若找不到放行且比 `dop853` 快的輸入，就帶數據回來問刪除；找到了，就提出保留方案。
使用者於 2026-11-15 以「1追認」確認保留與價格重校。

Family 去留與完整序列精度的規則由 [ADR-0078](../adr/0078-evolution-module-boundary.md#family-去留合格只比耗時) 擁有。
當時逐列放行的契約與價格校準說明見 [evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 及 [two-tone 價格模型](../../lib/zcu_tools/simulate/evolution/_two_tone_dispatch.py)。

## 問題

最初的量測只找到拒絕或單點較慢的例子，不能回答 `floquet` 是否值得保留。
拒絕前還要先建一週期 frame，錯誤的價格順序會讓這筆工作白費。

第一輪 spike 在單一 fluxonium 上查到，模態病態來自細緻平衡 population 區塊。
穩態 population 跨 12–16 個數量級。
分群正交化雖把 condition number 降到約 1，卻丟掉 bound 未計入的非對角項，不能當成修復。
這些數字來自第一輪 spike 的 condition 診斷，沒有使用使用者的 FTF 工作設定。

使用者也限制了研究來源。
2026-11-14 原話是「作为研究使用是可以用我的参数，但是实际进 report 的时候是不可以用的。」
兩輪 spike 曾唯讀使用者工作中的 FTF 設定。
使用者工作設定上的研究對照也有勝出區間，數字不入庫。
正式保留理由因此必須另找 committed 輸入，不能把模型量相近當成耗時也相同。

使用者同日也修正分流依據，要求價格、guard 與預判使用 qubit 實際感受到的量。
耦合要用當前 flux 的矩陣元素乘 gain，再考慮失諧與時間，不能直接按儀器 gain 分流。

## 考慮過的選項與比較

### 再研究一輪，確認批次是否能攤提固定成本

第二輪 spike 同時試了長 duration、批次掃描與詳細平衡的 `D^{1/2}` 對稱化。
單一 fluxonium 的對稱化沒有救回 admission。
它需要把換基底殘差與加權換算納入 bound，不能只看 condition number 變好。

最初保留提案只在使用者工作設定上有批次勝出數字。
審閱退回後，提案補做 committed 模型與 tracked 控制量的配對。
修訂後的保留區間縮成「列數多、duration 在 bound 內」，沒有沿用研究對照的倍數。

以下是「保留方案配對量測」的 committed 組。
模型取當時已提交的 FTF 設定，控制量取 [2026-10-18 Floquet 討論](2026-10-18-two-tone-floquet.md) 的 S9 描述。
每個輸入交錯比較 pinned、價格選路與預設路徑，各做三次，取 wall 中位數。
量測使用一個 worker、非獨占 slot，各組重複值約在 ±2% 內。
原始工作檔未入庫。

| 輸入 | 比較結果 | 當時能支持的結論 |
| --- | --- | --- |
| 11 點 detuning ±8 MHz、512 ns | pinned Floquet 7.24 s，DOP853 13.09 s，快 1.81 倍 | 批次攤提有勝出區間 |
| 11 點 duration 32–512 ns | 價格逐列分流 4.15 s，DOP853 6.86 s，快 1.65 倍 | 混合選路比整批釘死 family 有利 |
| 11 點 duration 520–1000 ns | 價格逐列分流 11.36 s，DOP853 18.92 s，快 1.67 倍 | 可保留合格列，其餘 fallback |
| 單點 512 ns | Floquet 3.17 s，DOP853 1.20 s | 不能由批次結果推單點也快 |

同組 committed 單點到了 1 µs 即被 bound 拒絕。
因此提案以約 0.9 µs 內的多列批次描述勝出區間，不把它當成通用時長 cutoff。
保留研究與後來的 tool-owned 校準使用不同輸入，兩組秒數不能混成同一次重測。

### 先重擬 DOP853，再校準兩邊價格

第一輪 spike 的 DOP853 校準格包含單一 fluxonium 的 13 點。
Designer 修正了最初「把實測能階冪次直接填入價格式」的建議。
價格式的頻寬項已隨能階增加，直接代入會重複計算維度效應。
隨後的 direct-estimate 重測採用基底 `3.47e-5`、`scale^0.324`，估價／實測為 0.926–1.095。
這組數字來自該次 direct-estimate 落地重測，不是後來的 FTF 校準格。

後續價格重校改用工具擁有的固定 [FTF_MODEL](../../tools/family_pareto_settings.py)，不讀使用者 YAML，也不新增依賴 lab 預設值的場景。
「價格重校量測」在 16 retained levels 上做十組控制，每組三次交錯配對，使用獨占計算。
每次新建 Engine 與 ModelSource，計時包住完整 evolve，不包含模型建構。
控制格涵蓋單點、detuning、duration、Stark 強度與載波、distinct-port balanced controls。

| 價格項 | 舊係數 | 重校係數 |
| --- | ---: | ---: |
| Floquet frame，16 levels、每 512 phase samples | 1.4 s | 4.18 s |
| Floquet probe-bin，16 levels | 1.05 s | 0.274 s |
| Floquet 每列，16 levels | 0.070 s | 0.008 s |
| DOP853 基底 | 3.47e-5 | 5.67e-5 |

表中數字全來自「價格重校量測」的落地報告。
Frame 與 bin 係數用八組放行控制的 process CPU 做 log-residual least squares。
每列係數固定為 0.008 s，實測 apply 約 2–11 ms。
DOP853 用十組 actual／old-estimate 比值的幾何平均調整基底。
這次把部分 bin 工作歸回共享 frame，沒有加入拒絕風險。

同輪 tool-owned 的 11 點 detuning 對照，Floquet wall 中位數為 3.33 s，DOP853 為 9.50 s。
價格比較因此改選 Floquet。
完整 benchmark 的五個 workload、三種 history mode、wall／CPU 共三十個比較，最大增幅為 +4.58%，沒有超過 10%。
這些都是價格重校的獨占量測，不是上表 committed 保留研究的重跑。

### 文件化逐列放行，延後三 family 排序

提案最初把 pinned 批次的一列拒絕誤讀成不支援逐列放行。
後續核對發現，預設 family 集合本來就保留已放行列，將拒收列交給其他候選。
固定 `families` 而沒有任何退路時，才會因一列無解而讓整次 evolve 失敗。
本次補文件與公開 Engine 驗證，沒有新增這項 admission 行為。

另一個選項是讓 `multitone_rwa`、`floquet`、`dop853` 全部按秒價格排序。
但 `multitone_rwa` 的精度 guard 尚待修正，也沒有可比較的秒價格。
當時選擇先處理 Floquet／DOP853 接縫，不把局部價格校準宣稱成完整預設集合已選到最快者。

本次遵循 [2026-10-18 Floquet 討論](2026-10-18-two-tone-floquet.md) 的「調價不調 guard」先例。
也遵循 [planner 討論](2026-10-18-planner-evidence-and-overhead.md) 的驗證邊界，用完整 benchmark 檢查退步，不以 stage 擬合取代整體證據。

## 被否決的選項與理由

- 立即刪除或只留下不會先試的備用實作。Committed 批次已找到勝出區間，刪除的停損條件沒有成立。
- 放寬 `0.95 * budget`。提案曾考慮把 0.95 提到 1.0 或約 1.9，兩者都缺少多段序列最終誤差證據。當時的預算分配是 omission 95%、frame／convergence 2.5%、數值餘量 2.5%。這些比例來自保留方案的預算核對；不能拿單段 bound 代替 ADR-0078 的完整序列標準。
- 單做 bath-only 提早停。保留方案的六個指定拒絕例中只有兩個可觸發，依階段計時估計各省約 25%，不是早停版本的實測加速。提案估計需 60–100 行，且救不到 committed 長時長的 probe 主導拒絕，當時未採用。
- 從 lab 預設檔建立持久 S9 場景。使用者於 2026-11-15 以「3對」確認不加。持久工具與測試不能依賴作者可能改動的設定，後來的價格校準改用工具擁有的固定模型。

## 留下的問題

- 三 family 排序仍等 `multitone_rwa` 修好與秒價格校準。完整預設集合當時仍先 offer 它。
- 價格重校的放行格中，Floquet 估價／實測仍為 0.50–1.56。Duration 批次低估，低 carrier 單點高估，這是價格重校報告留下的成本模型殘差，尚未做到估價逐格對上實測。
- 早期提案的階梯式拒絕風險項沒有落地。Single-fluxonium default 的八能階空白帶點也未補量，沒有足夠校準資料就不能把風險項當成已解決。
- 跨 evolve 的拒絕記憶當時沒有重算證據。後續只採用單次 evolve 的 exact-key frame 重用，見 [拒絕成本討論](2026-11-16-floquet-refusal-cost.md)。
- 放寬段落門檻仍須先有同型態多段序列的最終精度證據。

## 依據

本文整理兩輪 Floquet spike、保留方案修訂與審閱、direct-estimate 重測及價格重校報告。
原始工作檔未入庫，本文保留決策所需的結果與數字，沒有新增模擬量測。
使用者 FTF 工作設定的參數與衍生數字均未收錄。

- 第一輪 spike 提供 population 病態成因與不成立的正交化修法，第二輪提供對稱化與攤提研究。
- 「保留方案配對量測」提供 committed 批次表、時長邊界與 bath-only 階段估算。
- 「價格重校量測」提供工具固定模型的係數、3.33／9.50 s 對照、價格殘差與完整 benchmark。
- [ADR-0078](../adr/0078-evolution-module-boundary.md) 擁有 family 去留、最終精度與 preparation 壽命規則。
- [Evolution README](../../lib/zcu_tools/simulate/evolution/README.md) 與 [價格模型](../../lib/zcu_tools/simulate/evolution/_two_tone_dispatch.py) 保存當時已落地的契約與校準來源。
