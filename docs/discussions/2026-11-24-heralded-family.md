---
date: 2026-11-24
status: active
superseded_by: null
related:
  - docs/adr/0074-flat-readout-module.md
  - docs/adr/0072-module-boundaries.md
  - docs/discussions/2026-11-20-heralded-experiment-pruning.md
  - docs/discussions/2026-11-23-heralded-conditional-spectroscopy.md
  - docs/discussions/2026-11-23-heralded-pooled-display.md
  - zcu_lab/v2/heralded/README.md
basis:
  - "使用者 2026-11-24 裁決；架構核對與分析設計的原始工作檔未入庫。"
  - "repo@288738e803b85b6ab53232eaed449c4e279aa8ab：八個入口與共用模組的落地狀態。"
---

# 多類 herald 分析改用獨立家族

## 結論

使用者保留 `reset/herald` 和 `reset/mux_herald` 的單類篩選。需要保留多個類別的分析另開實驗。New Tab 新設與 `two_qubit` 並列的 `heralded`，模組也放在同名目錄。RB、conditional phase、conditional spectroscopy 搬入，另增 CZ amplitude、CZ optimize、Chevron、Local Z、Idle ZZ。規則見 [ADR-0074](../adr/0074-flat-readout-module.md#保留獨立-heralded-實驗的條件)，現行 key、Native tag 和各入口行為見 [heralded README](../../zcu_lab/v2/heralded/README.md)。

## 問題

2026-11-20 用「能否共用參數」判斷要不要保留獨立入口。2026-11-23 的條件頻譜需要比較四個實測初態，不估共用參數，已不符合這個門檻。另外，RB 和 conditional phase 原先只用 gg、ee 做共同估計，取數已保存的 ge、eg 未用在主分析。

「都存」起初被理解成讓普通實驗的 herald reset 交出所有分類。使用者更正：「如果是 reset herdering 的话，那就是真的就是只留一种。如果你想要全部保留，或是保留一种以上的，请独立的分化实验。」其中「herdering」「分化」指 heralding、heralded。普通入口只想篩選初態時，不需要看見 herald 讀出或改 Result。獨立實驗則要能比較被普通 reset 捨棄的類別。

## 考慮過的選項與比較

選單可保留各 `_heralded` 後綴、在 `two_qubit` 內開子選單，或另開與 `two_qubit` 並列的家族。另開家族還有兩種做法：只改選單，或連模組一起搬。使用者選後者：「我会选 C two，我认为表单的结构应该要跟实际结构对应，或者说一致，这样才好维护啊。」後續同意以模組目錄定 key，不把分組目錄藏起來。本次只搬 heralded；其餘 key 的整理另議。

使用者對分兩批新增五個入口答「那就采用建议吧。」CZ amplitude 在 gg 之外比較旁觀回返，並未把旁觀類加到首週擬合。CZ optimize 用四類計固定深度回返分數，不從單一深度估衰減。Chevron 將目標與三條旁觀圖分開，只擬合目標。Local Z 和 Idle ZZ 用 target herald e、spectator herald g 的資料補足主估計，把 spectator herald e 留作對照。由各實驗決定哪些類別用於同一個量，不為整個家族指定同一算法。

兩家族搬開後，有些機制仍要相同。例如 RB 共用 Clifford 臂生成與 depth 排程，兩版各自決定採集次序。兩家族各自直接 import 中立支援。可重用的數值擬合仍由 lib 持有；各 leaf 自行解讀計數、選模型適用的資料並作圖。這項共用位置由 designer 代裁，使用者未覆核；它未改動使用者選擇的選單與目錄結構，也未改 [ADR-0072 的依賴檢查](../adr/0072-module-boundaries.md#可執行投影)。

## 被否決的選項與理由

- 不讓 reset 新增「全留」模式或可見 herald 選項。篩選與保存多類在使用者選入口時就已分開。
- 不只移動選單，也不在 `two_qubit` 底下再分一層。前者讓 registry key 與程式目錄分離；後者讓普通入口與 heralded 入口仍隸屬同一家族。
- 不讓 `heralded` 直接 import 普通實驗，即使其 helper 改成公開名稱也會跨既有 leaf 邊界。不為兩家族增例外，也不複製 mux 驗證與 RB 排程。
- 本次不加 heralded joint readout 或 constructive phase。前者會重新定義分析量；後者的模型比較未顯示多類能提供新的相位選擇，沒有實機判據支持新增入口。
- 兩個四類共同擬合不採可用類子集。缺一類時，主估計的組成會改變。保留觀測與原因，不把 gg-only 或部分擬合當作四類結果。這是 designer 代裁，使用者未覆核。
- CZ amplitude 不採只在 CZ 前補碼、讀出後反標籤。這會讓 herald gg 改看 terminal ee，與普通量測的終端分類不同。若普通和 heralded 兩葉各做首週擬合，就得維護兩份權重、完整前綴與拒絕條件；上提到 lib 則把 Lab 的校準條件變成沒有其他 caller 的公共介面。首週擬合由中立的 [`support/cz_return.py`](../../zcu_lab/v2/support/cz_return.py) 持有。這是 designer 代裁，使用者未覆核。
- Chevron 不仿 CZ amplitude 在 drive 前後都加雙 Xπ，因為會多一段脈衝與等待。也不省末端 contrast，否則目標會改看 terminal ee。保留四類回返圖，只擬合 herald ee 的目標。這是 designer 代裁，使用者未覆核。
- CZ optimize 不將缺類填成零、不只平均可用類，也不按各類可用 seed 改權重。這些做法會讓不同候選的分數失去可比性。每候選也不跑多深度 RB，因為它會改變 CMA 的損失並增加取數，且與獨立 RB 驗收重疊。這是 designer 代裁，使用者未覆核。
- Idle ZZ 不沿用私有 multi-trace helper。直接引用會跨家族，搬其行為也會迫使各 trace 的衰減時間相同。普通和 heralded 兩葉不各寫一份 weighted fit，以免重複維護初值、界限與 covariance。新葉若改用舊 unweighted 入口，就不能用同一估計器與普通版對照。二段擬合會改變目標函式；另寫 profile-frequency solver 則要再驗搜尋與 covariance。這是 designer 代裁，使用者未覆核。

## 先例與仍適用的邊界

- 遵循 [herald reset 與逐 shot 遮罩](2026-11-20-herald-reset-shot-masks.md)。普通 reset 仍只留一類，也不把來源消耗的 herald 讀取交給實驗。
- 取代 [2026-11-20 pruning](2026-11-20-heralded-experiment-pruning.md) 以共用參數限制獨立入口的判準。準備污染仍可能改變擬合參數。特別是 Idle ZZ 並未把先例中會拉偏 ζ 的 spectator herald e 納入主估計。
- 遵循 [2026-11-23 條件頻譜](2026-11-23-heralded-conditional-spectroscopy.md) 的名目準備在前、以實測 herald 分類在後。現在它屬於家族規則，不再是共用擬合規則的特例。
- 遵循 [RB／conditional phase 合併呈現](2026-11-23-heralded-pooled-display.md) 的分組診斷與逐點計數權重。主圖從兩類擴成四類，不對合併點另做擬合。
- 遵循 [ADR-0069](../adr/0069-experiment-core-and-frontends.md) 的 Native 契約與 [ADR-0073](../adr/0073-measure-form-derivation.md) 的表單責任。新 key 只改開啟位置。同 schema 的舊結果用新入口載入，不更動原 tag 或增相容 reader。各 Form 仍自己選取來源與預設。
- 遵循 [ADR-0085](../adr/0085-numerical-fitting-owners.md)。共用數值函式不負責類別對齊、缺類資格、圖或寫回。這些決策留在 Lab。

## 留下的問題

同一批合成 counts 可以核對普通入口與 heralded 切片的數值，不能證明實機採集或準備等效。沒有量到正常工作量下的缺類率，或 RB 的 seed CI 因重抽樣不穩而缺席的頻率。其餘家族舊 key 的整理不在這次搬移內。

## 依據

2026-11-24 的使用者裁決、上述設計比較，以及八個實作與 guide。原始比較工作檔未入庫。固定量表只換三列的 `adapter`，不改 `scene_id` 與 overrides；舊報表保留原名稱。兩個既有 recipe 改名，未新增 spectroscopy recipe；現行操作名見 [recipes 指南](../../zcu_lab/recipes/two-qubit-guide.md)。
