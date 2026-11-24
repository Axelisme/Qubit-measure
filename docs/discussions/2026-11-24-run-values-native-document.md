---
date: 2026-11-24
status: active
superseded_by: null
related:
  - ../adr/0063-persistence-ownership.md
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0080-datafile-generic-container.md
  - 2026-11-24-cz-optimize-native-labber-format.md
basis:
  repo: 234bf075f043c1cb4e663ff9bf1686ad9772e1ac
  candidate: 6a3ea9e2d2c0d77b6eaff386753d95fc21605c55
  external: null
---

# Run 時固定值另存 Native 文件

## 結論

Run 時固定值另存於 `data.h5/docs/run_values`。實驗層用同一份保存宣告與 codec 把這些值寫入文件、還原為同一個 typed Result。缺文件等同缺欄，不另查舊變數。規則見 [ADR-0063〈Run 時固定值〉](../adr/0063-persistence-ownership.md#run-時固定值)。

## 問題

Mux Echo／Ramsey 的實際 detune 跟 Tone 一一對齊，但不隨 delay 改變。原寫法把它和從 cfg 複製的 carrier 都存成量測變數。單 qubit Echo／Ramsey 沒存實際 detune，重開後便無法使用它。CZ Optimize 把起點時長存成獨立變數；它只描述一次 Run 的起點，不是 evaluation 的一欄。分析也不能指望用同名資產還原當時的時長，因為資產可能移走或被換掉。

使用者先釐清現有位置各代表什麼：「既然datafile不負責解釋，那就不歸他管，目前每個欄位的實驗層慣例的語意為何？」聽完 `vars`、`docs/cfg`、`docs/metadata` 的分工後答「那就獨立文件吧」。對誰會讀檔，他說「我只會找data.h5」，並說可以「讓agent解析或是經Load看，前者是當資料格式不兼容時使用」。

## 比較過的選項

| 位置 | 能保留什麼，代價是什麼 |
| --- | --- |
| `docs/cfg` 或另一份 applied cfg | 真的讀回裝置設定時，能沿既有 cfg 欄位比對請求與實值。但實際 detune 是估值，CZ 起點時長沒有相同的 cfg 路徑。用 overlay 會混淆請求、設定和結果。 |
| `vars`、變數屬性或輔助座標 | 沿用變數可保存 unit、dtype 和 Tone 軸。變數屬性要為 CZ 選一個其實不擁有時長的變數；輔助座標可對齊 Tone，卻無法自然表示整次 Run 的單一時長。若把兩者加入公開 API，通用 datafile reader 也須更新。 |
| `docs/metadata` | 能沿用 Run 身分文件，也可不用擴標準欄而放自訂 key。但只想讀 run 身分的 caller 不需要實驗欄，兩份契約也不必一起演進。 |
| 根屬性、`groups` 或新增命名空間 | 根屬性夠放一個時長，逐 Tone 向量還要定另一種表示。不透明群組可保存帶 dtype 的大陣列，但本案是短向量與純量，沒必要另維護 HDF5 codec；新增命名空間還會擴大通用格式。 |
| 另一個檔案 | 可以獨立演進，但複製或搬動 `data.h5` 時容易漏帶第二個檔案，讀者也不再能只找 `data.h5`。 |
| 同檔的獨立 JSON 文件 | 純量和短的逐 Tone 向量已能用既有文件表示。通用 Container 保留文件；只有實驗 reader 需要懂欄位、單位與缺值。 |

選獨立文件沿用 [ADR-0080](../adr/0080-datafile-generic-container.md) 的通用容器與 ADR-0063 的保存權責。既有 [ADR-0069〈Native 可選變數與 Labber 投影〉](../adr/0069-experiment-core-and-frontends.md#native-可選變數與-labber-投影) 也將 Native 與 Labber 分開，但可選量測變數不是 Run 固定值。

## 未採用的選項

- **缺文件一律拒讀，或讀回舊變數。** 最初以一致拒讀避免逐實驗的相容分支。它會讓現有單 qubit 與 mux 的 Echo／Ramsey 檔無法 Load，連不需實際 detune 的擬合也做不了。使用者說「照讀或是拒絕選一個好維護的就好」，把選擇交給 designer：「都可以，你決定」。後來採用同一條缺欄規則。可缺欄成為 unknown，必填欄仍拒讀；舊 mux 的 `vars/actual_detunes` 不作備援，要查可直接讀原檔。[CZ 具名欄位的換版](2026-11-24-cz-optimize-native-labber-format.md) 只涉及當時的模擬測試資料，不將其拒讀做法外推到有實機資料的 Echo／Ramsey。
- **借 cfg／Container 版本，或由各 leaf 自訂版本。** Cfg schema、通用容器與固定值各有不同讀者；借用任一版號都會把無關的變更綁在一起。共用 v1 用整數版本，條目的 `unit` 和 `value` 放在一起。扁平格式會讓版本鍵與欄名共用空間，分表則要驗兩邊鍵一致。未知用 JSON `null`，不用非法的 JSON NaN，也不用可能是真值的 0。同版只驗宣告欄，typed 重存不保陌生欄；否則還要在 Result 外保留一份原文並決定衝突時以誰為準。
- **在離線匯入改用裸浮點比較 Tone，或推測沒存的值。** 歷史 mux 的 Tone 已經過 typed 整數映射，照原序比對能接受原規格的近整數，又不會放過交換順序。原 carrier log 仍須通過結構、有限值和 Tone 對齊驗證；不因它和 cfg 的數值不同而拒匯或覆寫 cfg。單 qubit 的歷史 log 沒有實際 detune，因此寫 `null`，不拿 nominal cfg 補。
- **沿用兩個既有 migration profile 的 revision。** 這次不等同於只修正 reader。新 writer 會產生固定值文件，mux 也不再保留兩個舊 log 為 Native variables。與[已證明等價的 MIST 轉換](2026-11-23-mist-legacy-labber-equivalence.md)不同，不能沿用 revision 讓新舊 writer 混在同一份 manifest。

## 留下的問題

不改寫舊 Native 檔，也不在正常 Load 加入 Labber fallback。四個既有歷史 importer 只使用有證據的欄位，保留原始 cfg、盤上座標與 Labber 證據，細節見[遷移說明](../../migrations/README.md)。歷史 carrier 和原 cfg 若不同，Mux Ramsey 的新頻率建議基準可能與舊畫面不同。這次不改 workflow aggregate 的格式或選集；若未來要存大量陣列，應另評估表示方式，不把它當作 v1 短向量。

## 依據

保存規則由 [ADR-0063](../adr/0063-persistence-ownership.md#run-時固定值) 擁有。作者接縫見[實驗層文件](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)，盤上與 typed 轉換由[共用 codec](../../lib/zcu_tools/experiment/run_values.py) 實作。v1 格式與四個歷史 Labber writer 方案由 liaison 依授權代答，使用者於 2026-11-24 以「都没有问题。」追認。原始調查與裁決工作檔未入庫；上文保留決策所需的取捨。
