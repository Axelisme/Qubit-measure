# Two-qubit recipes

**讀者：** operator、agent。設定 two-qubit recipe 的位置、掃描、herald reset 與校準寫回。

## 共用輸入

| 欄位 | 要填的值 | 省略時 |
| --- | --- | --- |
| `target` | `{qubit1, qubit2, coupler, mux}`。前兩項填不同的元件名，`coupler` 填保存這一對 CZ 校準的元件名，`mux` 填包含兩顆 qubit 的 mux group 名稱。 | 沿用目前 point 的 aliases。個別欄位省略或填 null 也沿用相應 alias。 |
| `reuse_tab_id` | 同 adapter 的閒置 tab ID。重用時先 Reset，不保留上次編輯。 | 開新 tab 並 Reset。 |
| `overrides` | 有順序的操作列表。每項填邏輯 `path` 和一個 `value`、`select` 或 `command`。 | 不做額外編輯。不接受 null。 |
| `reps`、`rounds` | 直接設定 Form 的 Reps 和 Rounds。 | 保留 Reset 值。 |

`target` 只改這次 Open／Reset 的 capture，不改 point aliases。Recipe 不交換兩個位置，也不從 qubit 猜 mux group。
表單不限制 qubit 的物理子種類或 coupler 的種類，仍要求 `group/mux` 並驗證[有序的 qubit／resonator 綁定](../v2/two_qubit/guide.md#綁定位置與讀出-tones)。兩個元件須不同且都在 `$mux.qubits` 中。已綁定和提交時的 readout 都須有完整 group 的 tone 數；操作員仍須核對 tone 順序。
本頁的 recipes 都需要 `$coupler`。實驗入口 `joint_readout`、`idle_zz` 與[兩個 conditional spectroscopy 入口](../v2/README.md#two-qubit-實驗清單)不需要。
使用舊 point 時，依[重綁步驟](../v2/two_qubit/guide.md#重綁舊-point)更新 aliases、CZ pulse library keys、module slots 和 `ext.cz_phase` keys，再開新 tab 或在 Run 頁按 Defaults。Notebook 內建的 `qubit1`、`qubit2` 與 GUI aliases 分開，不會自動互綁。

Reset 後先套具名欄位，再依序套用 `overrides`。要改引用脈衝的子欄位，先對脈衝路徑下 `command: to_inline`，再改子欄位。例如先轉換 `drive.qubit1`，再填 `drive.qubit1.gain`。
欄位仍有錯誤時，`check_ready` 回報要修正的路徑，recipe 不啟動 Run。

## 選用 mux herald

若 adapter 的 Reset 支援 `reset/mux_herald`，先在 library 準備該類型的模組。
Herald 與終端讀取用同一份完整 mux readout。模組的 `keep` 使用原始 tone 位置，不是 ADC 編號。
例如兩顆 qubit 位於第 0、1 個 tone 時，`keep: {'0': g, '1': g}` 只保留兩者都分類為 g 的 shot。

假設已存成 `CouplerA.reset.cz_herald`，用下列 override 選取實際 library key：

```json
{"overrides": [{"path": "reset", "select": "CouplerA.reset.cz_herald"}]}
```

`select: null` 清空可設為 null 的 Reset。選取 herald reset 不會切換 adapter。

## 回答寫回問題

Run 後執行 Primary Analysis。若分析有寫回候選，recipe 停在問答處，列出本次候選的值和目的地。
檢查圖與候選後，用 `answer` 回覆該問題的 `request_id`。接受時填 `accepted: true` 和選定的原樣 `items`。
略過時填 `accepted: false` 和空 `items`，不改校準。沒有候選就不詢問。

Recipe 只提交選定項目的 ID，並查寫回收據。`writeback_applied` 只在收據確認有成功項目時為 true。
完成 Run 後即使沒有擬合結果、略過寫回或寫回失敗，仍嘗試保存原始資料與已完成的產物。
用回覆的 `runs` 檢查各次 Save 的輸出與狀態。

## two_qubit_cz_amplitude

`two_qubit_cz_amplitude` 執行一次 `two_qubit/cz_amplitude`。
量測前依 [adapter 的準備說明](../v2/two_qubit/cz_amplitude/guide.md#preparation-and-inputs) 確認 Xπ 脈衝、兩路 CZ 脈衝與 mux readout。
`scale_range: [start, stop]` 對應 Form 的 Common CZ gain multiplier → Start、Stop（a.u.）。`scale_points` 寫入該掃描的 Expts；`reps`、`rounds` 分別對應 Reps、Rounds。
要估計第一個正回返，從 0 掃到涵蓋至少一個完整週期。
省略掃描欄位保留 Reset 值；Reset 沒有預填端點，需用具名欄位或 overrides 提供。

```json
{
  "target": {"qubit1": "Q1", "qubit2": "Q2", "coupler": "C0", "mux": "MG12"},
  "scale_range": [0, 1.5],
  "scale_points": 31,
  "reps": 1000,
  "rounds": 2
}
```

報告沿用 [第一個完整回返擬合](../v2/two_qubit/cz_amplitude/guide.md#fit-the-first-complete-return) 的欄位。
`scale` 是量測範圍內的第一個回返，`period` 是擬合週期，`error` 是週期的 1σ 不確定度。
`offset` 和 `contrast` 描述回返機率曲線，`fit_quality` 直接轉交擬合診斷。
無法給出回返時，`scale` 為 null，`reason` 說明原因；若仍有有限的擬合診斷，報告會保留。

有回返時，Primary 提出兩筆 CZ 脈衝 gain 更新和兩筆 coupler slot 更新。
脈衝和 slot 是獨立候選。需要設定 slot 時，請一併選取對應的脈衝與 slot。
檢查 [候選目的地與 Apply 規則](../v2/two_qubit/cz_amplitude/guide.md#preview-and-selected-apply) 再回答。

## two_qubit_local_z

`two_qubit_local_z` 執行一次 `two_qubit/local_z`。`target` 用 `qubit1`、`qubit2`、`coupler`、`mux` 綁定本次 Open／Reset。
量測前依 [adapter 的準備說明](../v2/two_qubit/local_z/guide.md#run-前) 檢查兩顆 qubit 的 Xπ/2 脈衝與 CZ 脈衝。
`gate_count` 對應 Form 的 CZ gates per shot (count)，填連續的非負整數，至少三個，例如 `[0, 1, 2, 3, 4]`。省略就保留 Reset 值；填 null 將 null 交給 Form 驗證。
`reps` 和 `rounds` 分別設定 Reps (count) 和 Rounds (count)。

報告的 `qubit1` 和 `qubit2` 各有 `intercept_deg`、`slope_deg`、`correction_deg`、`slope_error_deg`、`reason` 和 `fit_quality`。
`slope_deg` 和它的 1σ 不確定度 `slope_error_deg` 都按每個 CZ gate 計。`correction_deg` 是不重新 wrap 的負斜率。
頂層 `fit_quality` 為 null。任一角色擬合不可用時，該角色的 `reason` 說明原因，另一角色的結果仍會回報。

兩條擬合都給出 correction 才詢問是否寫入 `$coupler.ext.cz_phase`。候選的 `qubit1`、`qubit2` 是同次 Run 捕捉的元件名，兩個校正值是 `correction_qubit1_deg`、`correction_qubit2_deg`。
寫回不改脈衝或 slot。參見 [擬合解讀](../v2/two_qubit/local_z/guide.md#怎麼讀圖) 和 [候選內容](../v2/two_qubit/local_z/guide.md#preview-與-apply)。

## two_qubit_constructive_phase

`two_qubit_constructive_phase` 執行一次 `two_qubit/constructive_phase`。
依 [adapter 的準備說明](../v2/two_qubit/constructive_phase/guide.md#prepare-the-form) 確認兩路脈衝的共同頻率與掃描範圍。
`relative_phase_range: [start, stop]` 設定 Qubit2 relative to Qubit1 drive phase 的 Start 和 Stop (deg)。`duration_range: [start, stop]` 設定 Common waveform envelope duration 的 Start 和 Stop (µs)，兩端都必須大於零。
`relative_phase_points` 和 `duration_points` 分別設定兩軸的 Expts (count)。`reps` 和 `rounds` 分別設定 Reps (count) 和 Rounds (count)。
省略保留 Reset 值；Reset 未預填兩軸端點，須用具名欄位或 overrides 提供。

報告的 `relative_phase_deg` 是最大擬合速率對應的相對相位。找不到相位時，此欄為 null，`reason` 說明原因。
頂層 `fit_quality` 轉交速率平方的干涉擬合診斷，不回報個別脈衝長度曲線的擬合品質。

相位可辨識時，Primary 提出 Qubit2 脈衝的 phase 更新和 coupler 的 Qubit2 slot 兩筆候選。兩筆可獨立選取。
Qubit2 的新 phase 是來源 Qubit1 phase 加上擬合的相對相位。參見 [候選目的地與 Apply 規則](../v2/two_qubit/constructive_phase/guide.md#review-and-apply-phase-proposals)。

## two_qubit_conditional_phase

`two_qubit_conditional_phase` 執行一次 `two_qubit/conditional_phase`。
先依 [adapter 的準備說明](../v2/two_qubit/conditional_phase/guide.md#prepare-the-inputs) 確認兩顆 qubit 的脈衝與 mux 分類器。
`ramsey_role` 填 `"qubit1"` 或 `"qubit2"`，寫入 Form 的 Ramsey target，另一顆是 spectator。
Recipe 的 `target` 物件仍用來綁定 aliases。
`gate_count` 對應 Form 的 CZ gates per shot (count)，填至少兩個不重複的正整數，差值的 gcd 必須為 1。
`frequency_range: [start, stop]` 設定 Common CZ drive frequency 的 Start 和 Stop (MHz)，由低到高。`frequency_points` 設定該掃描的 Expts (count)。
`reps` 和 `rounds` 分別設定 Reps (count) 和 Rounds (count)。
省略具名欄位會保留 Reset 值。Reset 沒有預填頻率端點，需用具名欄位或 overrides 提供。

```json
{
  "target": {"qubit1": "Q1", "qubit2": "Q2", "coupler": "C0", "mux": "MG12"},
  "ramsey_role": "qubit2",
  "gate_count": [1, 2, 3],
  "frequency_range": [5799.8, 5800.2],
  "frequency_points": 21,
  "reps": 1000,
  "rounds": 10
}
```

`frequency_mhz` 是掃描範圍內唯一的 π 交點。沒有唯一交點時，`frequency_mhz` 為 null，`frequency_reason` 轉交分析的原因，不詢問寫回。
`fit_quality` 為 null。
有交點時，Primary 提出兩筆 CZ 脈衝頻率更新和兩筆 coupler slot 更新。
[脈衝與 slot 各自套用](../v2/two_qubit/conditional_phase/guide.md#select-a-frequency-and-apply)，回答前請確認兩路需要更新的項目。

## two_qubit_conditional_phase_heralded

`two_qubit_conditional_phase_heralded` 執行一次 `two_qubit/conditional_phase_heralded`。
使用同一組輸入欄位。Gate count 的間距須符合 [heralded adapter 的輸入條件](../v2/two_qubit/conditional_phase/guide_heralded.md#inputs-and-sequence)，例如 `[1, 2, 3]`。
每發都在相位序列前做 mux herald。兩個 conditional phase adapter 的 `reset` 只接受一般 reset，不能套用上文的 mux-herald reset override。

`frequency_mhz` 來自 [joint gg/ee 斜率的 π 交點](../v2/two_qubit/conditional_phase/guide_heralded.md#read-the-joint-conditional-phase)，不以只有 gg 的交點替代。
沒有唯一交點時同樣回 null 與 `frequency_reason`，不詢問寫回。
有交點時，候選同樣是兩筆脈衝頻率和兩筆 slot 更新。

## two_qubit_rb

`two_qubit_rb` 執行一次 `two_qubit/rb`，量測 reference 與 CZ-interleaved 兩臂。
按 [adapter 的準備說明](../v2/two_qubit/rb/guide.md#before-run) 校準脈衝、mux readout 和 local Z。
`target` 用 `qubit1`、`qubit2`、`coupler`、`mux` 綁定本次 Open／Reset。
Form 檢查 `$coupler.ext.cz_phase` 的元件名是否依序符合 `qubit1`、`qubit2`。不符就指出位置，符合才帶入兩個修正角度。

`depths` 設定 Random Clifford depths (count)，不含 recovery。
`seeds` 設定 Sequence seeds，兩臂共用 0 到 2147483647 的序列 ID。
`reps` 設定 Reps (count)，即每個 depth 每輪的 shots；`rounds` 設定 Rounds (count)，即每個 seed 和 arm 的採集輪數。
`reuse_tab_id` 指定要重用的 tab。`overrides` 在具名欄位設定後套用額外表單操作。
省略這四個量測輸入會保留表單值。明填 null 由 Form 驗證，不會改用預設。
兩份清單由 Form 去重並排序。範例：

```json
{"depths": [0, 2, 4, 8, 12, 16, 24, 32, 48], "seeds": [101, 102, 103, 104]}
```

### 長序列改用 const 波形

若長序列用光 waveform buffer，可在 `overrides` 把波形換成不佔 buffer 的 const。
先將 pulse 轉為 inline，再選 `Inline: const` 波形並設長度。以下在 tab 內換掉兩路 CZ 波形：

```json
{
  "overrides": [
    {"path": "drive.qubit1", "command": "to_inline"},
    {"path": "drive.qubit1.waveform", "select": "Inline: const"},
    {"path": "drive.qubit1.waveform.length", "value": 2.0},
    {"path": "drive.qubit2", "command": "to_inline"},
    {"path": "drive.qubit2.waveform", "select": "Inline: const"},
    {"path": "drive.qubit2.waveform.length", "value": 2.0}
  ]
}
```

`length` 的單位是 µs。兩路 CZ 必須同長；換波形後，請用重校的 gain 和 local-Z 修正。
如需更換 X90 或 X180 波形，同樣對 `x90_qubit1`、`x90_qubit2`、`x180_qubit1` 或 `x180_qubit2` 的 pulse 路徑和其 `.waveform` 操作。
這些覆寫不改 library。

報告的 `gate_error` 是 Analyze 的 CZ error estimate，`error_ci_low` 後接 `error_ci_high` 是 95% seed CI 的上下界。這些值與 `p_reference`、`p_interleaved`、`reference_epc` 都是無單位比值。
`n_paired_seeds` 與 `bootstrap_successes` 分別是配對 seed 數和有效 bootstrap fit 數，單位為 count。`warning` 交代模型限制。
用 [adapter 的報告判讀](../v2/two_qubit/rb/guide.md#interpreting-the-result) 檢查誤差和兩臂的衰減。
兩臂診斷各在 `reference_fit_quality` 和 `interleaved_fit_quality`。基底 `fit_quality` 為 null。
沒有估計時，估計欄位為 null，`reason` 說明原因。分析拋錯時，執行回失敗與原錯誤。
完成 Run 後即使分析失敗也保存。不詢問寫回。

## two_qubit_rb_heralded

`two_qubit_rb_heralded` 執行一次 `two_qubit/rb_heralded`。
每個 shot 自帶 [herald readout](../v2/two_qubit/rb/guide_heralded.md#shot-sequence)。
`reset` 只接受一般 reset，不選 `reset/mux_herald`。
沿用 [RB 的輸入](#two_qubit_rb)。長序列也可用上面的 [const 波形覆寫](#長序列改用-const-波形)。

估計欄位和失敗原因與 RB 相同，但衰減來自 [gg/ee 聯合擬合](../v2/two_qubit/rb/guide_heralded.md#joint-decay-and-uncertainty)。
`n_paired_seeds` 計算兩組有效 seed 的聯集。
四欄診斷依序是 `gg_reference_fit_quality`、`gg_interleaved_fit_quality`、`ee_reference_fit_quality`、`ee_interleaved_fit_quality`。
基底 `fit_quality` 為 null。完成 Run 後即使分析失敗也保存。不詢問寫回。
