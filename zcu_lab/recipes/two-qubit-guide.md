# Two-qubit recipes

**讀者：** operator、agent。設定 two-qubit recipe 的角色、掃描、herald reset 與校準寫回。

## 共用輸入

| 欄位 | 要填的值 | 省略時 |
| --- | --- | --- |
| `target` | `{f1, f2, coupler, mux}`。F1 與 F2 填兩顆 fluxonium 的元件名稱，`coupler` 填 transmon 名稱，`mux` 填包含兩顆 fluxonium 的 mux group 名稱。 | 沿用全域 aliases。個別角色省略或填 null 也沿用相應的全域 alias。 |
| `reuse_tab_id` | 同 adapter 的閒置 tab ID。重用時先 Reset，不保留上次編輯。 | 開新 tab 並 Reset。 |
| `overrides` | 有順序的操作列表。每項填邏輯 `path` 和一個 `value`、`select` 或 `command`。 | 不做額外編輯。不接受 null。 |
| `reps`、`rounds` | 直接設定 Form 的 Reps 和 Rounds。 | 保留 Reset 值。 |

角色只改這次 Reset 的 capture，不改全域 aliases。Recipe 不交換 F1/F2，也不從 qubit 推測 mux group。
Reset 後先套具名欄位，再依序套用 `overrides`。例如 `drive.f1.gain` 是脈衝 gain 的邏輯路徑。
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
`scale_range: [start, stop]` 填無單位的共同 gain 倍率範圍。`scale_points` 寫入 Form 的 Expts。
要估計第一個正回返，從 0 掃到涵蓋至少一個完整週期。
省略掃描欄位保留 Reset 值；Reset 沒有預填端點，需用具名欄位或 overrides 提供。

```json
{
  "target": {"f1": "F1", "f2": "F2", "coupler": "C0", "mux": "MG12"},
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
