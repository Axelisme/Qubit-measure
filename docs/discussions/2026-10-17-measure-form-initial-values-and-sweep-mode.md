---
date: 2026-10-17
status: active
superseded_by: null
related:
  - docs/adr/0065-cfg-editing.md
  - docs/adr/0068-remote-transport.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/discussions/2026-10-13-measure-form-redesign.md
  - zcu_lab/v2/gui-contracts.md
  - zcu_lab/v2/_support/measure/README.md
basis:
  - "repo@4b5385843：Measure Form UX 與校準初值政策落地"
  - "repo@72fd2f458：sweep 欄位 range／center 與 expts／step 切換落地"
  - "repo@6848f3217：撰寫時的 main HEAD"
---

# Measure Form 的校準初值與 sweep 模式切換

## 結論

使用者對 Form 初值的總原則是「可預測與一致比正確重要」：

- 缺值：用公允值，Form 照常打開。
- 存在但無效：留空，由操作員看見並補上，Form 仍要能打開。
- 來源故障（alias、library 錯誤）：照原錯誤報出。
- 初值在打開 Form 時算成數字填入。不在欄位裡放引用校準參數的長 expression。
- 同一條公式到處一致。算出的值不合理時，讓後續驗證報錯，不在 Form 裡 clamp 或暗中修正。

已定的具體數值：

| 情境 | 初值 |
| --- | --- |
| T1 缺值 | 等待時間以 30 µs 為尺度 |
| T1 無效 | Form 打開，sweep stop 留空 |
| Probe pulse 長度 | 1 µs，不看 x180 |
| Amplitude Rabi 的 gain 掃描 | −0.3～1.0 |
| Amplitude Rabi 的 pulse 長度 | 1.5 × x180 長度 × x180 gain；沒有 x180 時 1.0 µs |
| Length Rabi 的掃描上限 | 4 × x180 長度；長度缺值時 1 µs |
| Lookback 沒有 time_of_fly | trig_offset 0 µs，ro_length 1.5 µs |
| Lookback 有 time_of_fly | trig_offset = time_of_fly − 0.1 µs，結果為負也照樣帶入 |
| 弛豫類等待時間 | 5 × T1；JPA SNR 類 1 × T1 |

有效性邊界：

- x180 長度：有限正數。
- x180 gain：有限數，且 0 < gain ≤ 1。負的 gain 讓長度留空，不改用 |gain|。
- gain 為 NaN 或 Inf：gain 與長度一起留空，因為 JSON 無法表示這兩種值。
- time_of_fly：有限實數且 ≥ 0。

x180 存在但缺長度或 gain 時，長度留空，不當成「沒有 x180」而套用 1 µs。x180 是 arb waveform 時，Amplitude Rabi 沿用 x180、不改長度：arb 沒有可編輯的長度欄位，初值也不讀 asset。

Sweep 欄位支援右鍵切換 range／center 與 expts／step，數值自動換算。換算後參與的 expression 變成數字，不保留原式。

## 問題

1. Measure Form 的初值以前要操作員手填，或沿用各實驗各自的習慣。缺值、無效值與來源故障的處理方式不一致，同類實驗的初值也不一致。
2. Sweep 欄位只有 start／stop／點數一種表示。以校準頻率為中心掃描，或固定步長掃描時，操作員要自己換算。

## 考慮過的選項與比較

### 缺值與無效值

使用者第一次面對「沒有 T1」時說：「我認為沒有t1可以採取的行為是：不填，或是一個公允值，比如100us」，接著選定「公允植吧，但取30us好了」。之後的 probe、Rabi 與 Lookback 都沿用「缺值給公允值」。

無效值另行處理：值存在但不合理，代表來源有問題，應該讓操作員看見，所以留空。這要求 `cfg_model` 的 `Range` 端點能表示「留空必填」。採用的做法是端點允許 `None`，只在實驗宣告層。GUI 的 `LinRange` 與硬體的 `SweepCfg` 仍只收有限數字，提交前就擋下空值。

Lookback 的 time_of_fly 小於 0.1 µs 時，trig_offset 會算出負數。比較過兩種做法：clamp 到 0，或照樣帶入讓後續驗證報錯。使用者選後者，理由是「可預測與一致比正確重要」。

### Pulse 來源

Probe 不看 x180。使用者的理由是「probe不看x180/pi_len,因為通常比後者長很多，我認為取1us就好」。

Amplitude Rabi 的長度原本只考慮「有 pi 長度就用它」。使用者改成「如果有pi_len，取1.5*pi_len_gain_product的長度，沒有的話就1.0us」：用面積相近的 pulse 掃 gain。

兩種 Length Rabi 原本來源不同：Singleshot 用 qub_probe，Twotone 用 x180。統一成有 x180 就用 x180，與 Amplitude Rabi 一致。

使用者曾以為 pi_len／pi_gain 是獨立參數。實作讀的是 `$qubit.module.x180` 的 `waveform.length` 與 `gain`，這兩個名字只是區域變數。使用者也表示「我傾向初始值直接是數值」，現行實作已在打開 Form 時算出數字。

### 共用政策放在哪裡

`zcu_lab/v2/gui-contracts.md` 原本規定具體數值只能寫在各實驗的 `form.py`。T1 的 30 µs 與 Rabi 的倍率被多個 Form 共用，散寫容易改一處漏一處。決定是：已定案、而且確實被多個 Form 共用的初值政策，可以集中在具名的 domain helper；單一實驗的政策仍留在 `form.py`。使用者回覆「同意，這判斷標準沒問題」。

### Library 引用中的 waveform

x180 的 waveform 若是 library key，「複製 pulse 再改長度」原本會失敗。決定由 Form adapter 擴充既有的 `Slot(copy=True).set`：只把路徑上需要的子 Spec 複製成 inline，library 本身不變。不讓 Rabi 自己拼 cfg。

### Sweep 模式

數值契約：

- Step 除不盡時保留 step，點數 = floor(|span| / |step|) + 1，並重算端點。
- Span 帶正負號，10→4 等於 center 7、span −6。
- 切換要嘛整個成功，要嘛整個不動。

模式存放位置比較了兩種：

| 選項 | 結果 |
| --- | --- |
| 只在 Qt widget 切換 | MCP 與畫面看到不同欄位；center 存不了 `$qubit.freq` 這類 expression |
| 存在 Form 共用的輸入狀態（採用） | GUI、MCP、tab 還原看到同一個模式；硬體 cfg 與資料檔格式不變 |

Recipe 端的取捨：`RecipeTab.set_sweep` 與高階 recipe 仍用 start／stop／點數一次設定完整掃描，並把模式重設成 range + expts。好處是呼叫方不用知道 GUI 當下的模式。代價是 GUI 選好的 center 模式與 expression 會被 recipe 覆蓋。低階欄位編輯要先讀 description，才知道當下有哪些欄位可寫。遠端切換模式用通用的 `tab.run_cfg_command(tab_id, path, name)`，不新增高階 MCP tool。

使用者接受上述取捨，GUI 走查跳過驗收：「通過，GUI跳過驗收，直接land然後封存」。

## 被否決的選項與理由

- 缺值一律留空：首次操作多一步，與使用者偏好公允值不一致。
- 無效值 fallback 到公允值：把來源問題藏起來。
- 負 gain 取絕對值推導長度：改動使用者給的公式，也破壞一致性。
- trig_offset 為負時 clamp 到 0：在 Form 裡暗中修正數值。
- arb x180 改成 const 或讀 asset 推長度：改變物理意義，或違反「初值不讀 asset」。
- 新增「缺值」sentinel 型別：只有一個用途，直接用 `None`。
- Sweep 不整除時一律報錯：擋掉最常見的用法。
- Sweep 專用 RPC，或擴充 `edit_cfg` 混入 command：每多一種切換就要再加分支，或批次語意變複雜。

## 留下的問題

- Recipe 會覆蓋 GUI 選的 sweep 模式。若之後需要「recipe 只寫數值、保留模式」，要另開一張票。
- 既有的 `to_inline` 專用入口還沒遷移到通用 command RPC。
- arb waveform 若要能調長度，屬於 arb 的 domain 能力與 UI 設計，未處理。

## 依據

本快照依據 2026-10-16～17 的 advisor 代裁與使用者審閱。原始問答與決策檔在本機工作目錄，未入庫。上表的數值與規則已寫在 `zcu_lab/v2/_support/measure/README.md`、`zcu_lab/v2/gui-contracts.md`、`zcu_lab/v2/measure-authoring.md`、`lib/zcu_tools/cfg_model/README.md` 與 ADR-0065／0068／0073；兩者不同時以這些文件為準。
