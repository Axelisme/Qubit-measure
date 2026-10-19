# 不用遷移腳本的手動遷移指南

本指南把舊 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/` 整理成新 entry。
範例直接建立同根下的 `result/<name>/` 與 `Database/<name>/`，不必先放另一個 root 再搬回。
新格式根目錄統一為 `result/`，與 [GUI runtime](../lib/zcu_tools/gui/session/services/build.py) 一致。
他機已寫入 `results/<name>/` 時，依[根目錄搬移](#將他機已寫入的-results-搬到-result)處理，不必重做遷移。
`result_root` 可明確指定其他目錄，名稱本身不決定格式。
也適用於他機應急工具只遷一半的資料，見[部分遷移接手](#從部分遷移的狀態手動接手)。
本文不要求執行 `migrate_storage`，也不宣稱自動備份、exps discovery、predictor 轉換或 cleanup 已可用。
腳本操作另行補充。舊腳本可能仍限制在整個舊 result root 內寫入，本文不保證其 CLI 已支援同根目的地。
手動操作只要求新 entry 與本次兩個來源樹不重疊。**先驗證完整備份，最後才清理舊來源。**

格式依據是 [entry owner](../lib/zcu_tools/resources/entry/README.md)、
[context owner](../lib/zcu_tools/resources/context/README.md)、
[持久保存 ADR](adr/0063-persistence-ownership.md)、
[cfg ADR](adr/0065-cfg-editing.md)、[module 分組 ADR](adr/0072-module-boundaries.md#library-分組與-module-slot)
與 [readout ADR](adr/0074-flat-readout-module.md)。
[舊 mapping](../migrations/storage_mapping.py) 可查歷史欄位，但其中固定的 R1/Q1/J1 是那個 profile 的命名，不是所有資料的拓樸。
不要把 `Q0` 一律改成 `Q1`。正常 loader 不會替你轉換舊格式。

## 1. 固定來源、目的與授權

停止會寫入這些目錄的 GUI、Notebook 與其他 agent。取得本次讀取、寫入與最後刪除的授權。
未取得刪除授權時，可完成新資料與報告，但停在清理之前。下列命令使用 Bash、GNU coreutils 與 findutils。
所有範例路徑與數字都必須換成該機已確認的資料，不要直接貼到真資料上執行。

```bash
set -euo pipefail
export REPO=/absolute/path/to/Qubit-measure-gui
export CHIP=TestChip QUB=Q0 NAME=migrated-Q0
export OLD_R="$REPO/result/$CHIP/$QUB"
export OLD_D="$REPO/Database/$CHIP/$QUB"
export RESULTS="$REPO/result" DATABASE="$REPO/Database"
export NEW_R="$RESULTS/$NAME" NEW_D="$DATABASE/$NAME"
export BACKUP="$(dirname "$REPO")/$NAME-backup"
export WORK=/absolute/path/outside/repo/migration-Q0-review
mkdir -p -- "$WORK"
printf '%s\n' "$OLD_R" "$OLD_D" "$NEW_R" "$NEW_D" "$BACKUP" "$WORK"
```

行動與完成判斷：

- 用 `realpath -m -- <path>` 核對上述路徑。BACKUP 必須是 repo 外獨立的 `<name>-backup`，不能已存在，不能包住來源。WORK 也放 repo 外，不能在舊來源內。
- 允許新舊 entry 共用 result/Database 根目錄，但新 entry 不得寫入或包住本次任一舊來源樹，也不得透過 symlink 重疊。CHIP、QUB、NAME 必須是已確認的單一路徑段，不含 `/`、`.` 或 `..` 路徑段。部分接手也要檢查新 entry 內的 symlink；不能讓 figures、records 或其他目的子目錄跳出新根目錄。
- 全新遷移用 `test ! -e "$NEW_R" && test ! -e "$NEW_D"`，預期 exit 0。已有任一目的地就改走部分接手，不刪掉它重來。
- 用 `find "$OLD_R" "$OLD_D" -type l -print` 檢查連結，再用 `find "$OLD_R" "$OLD_D" ! -type d ! -type f -print` 檢查特殊檔。預期皆無輸出。若有，先停下確認實體來源及保全方式；不要跟隨連結複製或清理。
- OLD_D 若原本不存在，記錄實際缺失與使用者確認，不建立空 Database 假裝已備份。以下雙來源備份範例假設兩個來源都存在。

完成時，把路徑、操作者、授權範圍與當前 checkout commit 寫入 `$WORK/notes.md`。
預期得到一份本次操作紀錄，而不是工具 manifest。

## 2. 完整備份 result 與 Database，逐檔驗證

備份範圍是這次的兩個完整來源目錄，包括未知檔、隱藏檔與空目錄，不只挑已知格式。
空間不足、讀取失敗或任一 hash 不同，都不進入後續遷移。

```bash
hash_tree() {
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z |
    xargs -0 -r sha256sum)
}
test -d "$OLD_R" && test -d "$OLD_D"
test ! -e "$BACKUP" && test ! -L "$BACKUP"
hash_tree "$OLD_R" > "$WORK/result-before.sha256"
hash_tree "$OLD_D" > "$WORK/Database-before.sha256"
mkdir -- "$BACKUP"
mkdir -p -- "$BACKUP/result/$CHIP" "$BACKUP/Database/$CHIP"
cp -a -- "$OLD_R" "$BACKUP/result/$CHIP/$QUB"
cp -a -- "$OLD_D" "$BACKUP/Database/$CHIP/$QUB"
export BR="$BACKUP/result/$CHIP/$QUB" BD="$BACKUP/Database/$CHIP/$QUB"
cp -- "$WORK/result-before.sha256" "$BACKUP/result.sha256"
cp -- "$WORK/Database-before.sha256" "$BACKUP/Database.sha256"
(cd "$BR" && sha256sum -c "$BACKUP/result.sha256")
(cd "$BD" && sha256sum -c "$BACKUP/Database.sha256")
hash_tree "$BR" > "$WORK/result-backup.sha256"
hash_tree "$BD" > "$WORK/Database-backup.sha256"
cmp "$WORK/result-before.sha256" "$WORK/result-backup.sha256"
cmp "$WORK/Database-before.sha256" "$WORK/Database-backup.sha256"
diff -qr -- "$OLD_R" "$BR"
diff -qr -- "$OLD_D" "$BD"
hash_tree "$OLD_R" > "$WORK/result-after-backup.sha256"
hash_tree "$OLD_D" > "$WORK/Database-after-backup.sha256"
cmp "$WORK/result-before.sha256" "$WORK/result-after-backup.sha256"
cmp "$WORK/Database-before.sha256" "$WORK/Database-after-backup.sha256"
```

預期每個檔案顯示 `OK`，所有 `cmp`、`diff` exit 0，且沒有差異輸出。
比較完整 hash 清單也會發現額外檔案；只跑 `sha256sum -c` 不能發現多出的檔案。
將結果存入 WORK。完成後以 BR、BD 作為後續唯讀輸入，保留舊來源到最終核對。
若流程中斷，不覆蓋既有 BACKUP；先依保存的清單重驗。驗證失敗的半份備份不能作為清理依據。

若資料已被應急工具搬走，現在複製的殘存目錄不是「原始完整備份」。
必須找回搬移前的完整備份，或用可信的原始盤點/hash 清單與新舊兩端逐檔重建完整備份。
從新位置補回時，放回備份內的原始相對位置並比對原始 hash，記錄補回來源。
缺少原始清單、檔案或完整性證據時，記為阻擋，不授權清理。

## 3. 盤點每個來源的去向

```bash
find "$BR" -type f -printf '%P\n' | LC_ALL=C sort > "$WORK/result-files.txt"
find "$BD" -type f -printf '%P\n' | LC_ALL=C sort > "$WORK/Database-files.txt"
find "$BR/exps" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
find "$BR/exps" -type f -iname '*.png' -printf '%P\n' | LC_ALL=C sort
```

若沒有 exps，第三、四行改查實際存在的舊直屬 point 目錄，不憑空建立。
預期列出 point label 與 PNG 相對路徑。每個來源至少在報告中有一列。

| 舊檔 | 新位置與處理 |
| --- | --- |
| 根層 `params.json` | 原檔可原樣存 `result/<name>/params.json` 作稽核副本；`fluxdep_fit` 另寫 `result/<name>/predictors/<component名>.json`，其餘依第 4 節保存到 component 的 `ext.legacy_params`，不能只 copy 就算完成。 |
| 根層 `meta_info.json` | 依欄位 mapping 寫 `setup.yaml`；與 params 衝突時 meta 優先。 |
| `exps/<sub>/meta_info.json` | `result/<name>/points/<sub>/point.yaml`，先繼承完整 setup 再套用 point meta。 |
| `exps/<sub>/module_cfg.yaml` | 同一 point 的 `module_cfg.yaml`，轉正式 Spec，不原樣冒充新格式。 |
| `exps/<sub>/image/*.png` | 同一 point 的 `figures/<原檔名>`，不加 run_id。 |
| 舊直屬 `<sub>/{meta_info.json,module_cfg.yaml}` | 同名新 point。若與 `exps/<sub>` 重名，停下列出兩個來源，不合併。 |
| 根層 `module_cfg.yaml` | 未證明 point 歸屬就保全並 pending，不造 setup library。 |
| `samples.csv` | 原樣保全到 `Database/<name>/migration-preserved/result/samples.csv`；不是新的 SampleTable。 |
| `autofluxdep_runs/**` | 同相對路徑保全到 `Database/<name>/migration-preserved/result/autofluxdep_runs/**`；不是新 workflow run。 |
| exps 外的 `image/**`、未知 result 檔 | 原樣保全到 `Database/<name>/migration-preserved/result/<相對路徑>`，無法確定關聯則 pending。 |
| `result/<chip>/<qub>/arb_waveforms/**` 或 `Database/<chip>/<qub>/arb_waveforms/**` | 原樣複製到 `Database/<name>/arb_waveforms/**`，保留 arb_waveforms 下的相對 asset key 並核對 hash；兩來源對應同一目的但 bytes 不同時停下列衝突，不覆寫。 |
| Database Labber 檔 | 依第 7 節核對必要證據，再產 native 2.1 並保留原 Labber bytes。未知開始時間用 None/unknown，缺可信 ID 用原生 API 分配。其他必要證據不足仍 pending，可另存 `migration-preserved/Database/<原相對路徑>`。 |
| 其他 Database 檔 | 備份保全，報告記明原相對位置與未知用途；需新副本時放 `migration-preserved/Database/`，不冒充 run。 |

不要從 `070716_0.500` 解讀磁通或時間。這類 label 可原樣使用。
不合法 label 需使用者決定轉換規則，記錄舊名、新名與理由後再做。

### 手動報告的最小格式

在 WORK 建 `manual-report.json`。這是交接清單，**不是** `migration-state.json`，不能拿去 `--resume`。
以下只示範一筆永久 pending，其餘逐檔及逐欄增加；hash 欄位填實際 SHA256。

```json
{
  "mode": "manual-handoff",
  "entry": "migrated-Q0",
  "backup": "/external/migrated-Q0-backup",
  "items": [
    {
      "source": "result/TestChip/Q0/samples.csv",
      "source_sha256": "<actual SHA256>",
      "destination": "Database/migrated-Q0/migration-preserved/result/samples.csv",
      "status": "permanent-pending",
      "location": "rows and coordinate units",
      "reason": "No verified row-to-point and coordinate mapping",
      "backup_path": "result/TestChip/Q0/samples.csv"
    }
  ],
  "unfinished_publications": [],
  "temporary_pending": []
}
```

`verified` 表示目的已讀回並核對。`temporary-pending` 表示仍能完成但尚未完成，例如等 point 的圖。
`permanent-pending` 表示目前無法安全轉換，已在完整備份中定位並說明缺哪項證據。
衝突列出兩個來源、值、單位與採用理由。完成判斷是每個檔與每個未轉欄位都有去向，不能只記總數。

## 4. 分開保存 predictor 與 setup

先確定舊 qubit 對應哪個新 component，以及物理 kind。
全新 entry 可沿用已確認的 `Q0` 等名稱；部分接手沿用現有 mapping 與 component 名稱。
找不到唯一對應時，整份 params 列 pending，原檔留備份，不任選一個 qubit。

全新條目可用公開 API 建骨架，避免手填 UUID。這裡呼叫資源 API，不是遷移腳本，也不接硬體。

```bash
uv run --directory "$REPO" --no-sync -- python - <<'PY'
import os
from zcu_tools.resources.entry import ResultEntry
entry = ResultEntry.create(
    os.environ["NAME"], result_root=os.environ["RESULTS"],
    database_root=os.environ["DATABASE"],
)
print(entry.entry_id, entry.result_path, entry.database_path)
PY
```

預期印出新 UUID 與兩個目的路徑，產生 setup、points、records 及同名 Database 目錄。
既有條目不跑 create，不重新產生身分。保存此 UUID 與建立時間。

### fluxdep_fit 寫成 entry predictor

`fluxdep_fit` 不再轉成 component 欄位，也不隨 setup 複製到 points。
將它寫成 `$NEW_R/predictors/<component名>.json`，例如已確認唯一對應為 Q0 時用 `Q0.json`。
component 名只決定本次遷移的檔名，不決定 predictor 的模型或參數 schema。
格式 owner 是 **predictor-file**。其實作尚未 land 時，可按以下格式用編輯器直接寫 JSON。
land 後改用 `resources.predictor_file.save_predictor_file(Path, PredictorFile)`，型別為 `PredictorFile`、`PredictorSource`。
`ResultEntry.predictors_path` 只回傳路徑，不建立目錄，不能拿屬性讀取成功當成檔案已存在。
這是手動寫法，不表示舊 migration 腳本已能輸出 predictor，也不保證目前 predictor UI 已能載入。

先從唯讀備份核對模型、參數及來源，再建立 `predictors/` 並寫入檔案。
目的檔已存在時先走部分接手，只核對或補缺，不覆寫不同內容。
最小格式包含以下欄位。頂層只允許表列的六個 key。
`source` 只允許 `run_id`、`created_at`，不要另加 metadata 或 migration 註記。

| 欄位 | 寫法與證據 |
| --- | --- |
| `format`、`version` | 固定為 `"zcu.predictor"`、嚴格整數 `1`，不接受 `true` 或 `1.0`。不是 parameter-container 的 `format_version`。 |
| `model` | 格式接受 `"fluxonium"` 或 `"transmon"`。本次只轉換下述已有契約的舊 Fluxonium fit，不從 component kind 或名稱猜測。 |
| `params` | 模型參數及 `flux_half`、`flux_period`、`flux_bias` 的 mapping。沿用 simulate 的 `Params.from_mapping`，只用該模型的合法欄名；未知值可省略或填 `null`，已知值須是有限數字，不接受 bool、數字字串、NaN 或 Infinity。完整性與物理約束由 `build_predictor` 檢查。 |
| `flux_unit` | 必填，非空字串或 `null`。只接受 params 根層 `flux_unit` 或 `fluxdep_fit.flux_unit` 的明示單位。只有一處時採該值，兩處都有時必須相同；皆無或衝突時寫 `null` 並列 pending。數值不換算。 |
| `source` | `run_id`、`created_at` 都必填，值為非空字串或 `null`。舊 fit 的 `run_id` 寫 `null`；`created_at` 只採用可證實為含時區 ISO 8601 的 `fluxdep_fit.timestamp`，否則寫 `null` 並列 pending。不能拿 entry UUID、檔名、mtime 或本次遷移時間代填。 |

舊 [FluxDepFit 契約](../lib/zcu_tools/resources/qubit_params.md) 的手動 mapping 如下：

| 舊值或既有預設 | predictor 位置 |
| --- | --- |
| `fluxdep_fit.params.EJ`、`EC`、`EL` | `model` 寫 `"fluxonium"`，數值分別寫 `params.EJ`、`params.EC`、`params.EL`，保留 GHz。 |
| `fluxdep_fit.flux_half`、`fluxdep_fit.flux_period` | 分別寫 `params.flux_half`、`params.flux_period`，不換算。 |
| 舊 predictor 的 bias 預設 | `params.flux_bias` 寫 `0.0`。這是舊契約的預設，不是從 `flux_int` 或 half/period 推算。 |
| `fluxdep_fit.flux_int` | 只作舊 alignment，不寫入 predictor，也不改名成 `flux_bias`。原文留備份。 |
| `fluxdep_fit.timestamp` | 符合上述時間證據才寫 `source.created_at`，例如 `"2026-10-19T08:00:00Z"`；否則 `null` 並在報告定位備份內原文或缺失。 |

已知 mapping 但缺參數值時，可以省略該參數或填 `null`，例如 `"EC": null`，並列 pending。
JSON 格式有效不等於參數已足夠建立 predictor，不用零值補缺，也不為未知 source 造值。
舊 owner 未定義的 transmon 或帶明示 `model` 的 fit 保持 pending，不自行擴充 legacy 輸入契約，也不把 EJ 改名成 EJmax。
其他映射若缺可靠依據，保全原文及 metadata，報告記錄缺項與備份位置，請使用者與格式 owner 核對。

以下是合成的舊 Fluxonium fit 遷移範例，不是 TestChip/Q0 的實測參數。
假設允許的單位來源明示 A，舊 fit 的數值已核對，但沒有可證實的 timestamp，`result/<entry>/predictors/Q0.json` 可寫成：

```json
{
  "format": "zcu.predictor",
  "version": 1,
  "model": "fluxonium",
  "params": {
    "EJ": 6.27,
    "EC": 1.41,
    "EL": 0.80,
    "flux_half": 0.001,
    "flux_period": 0.002,
    "flux_bias": 0.0
  },
  "flux_unit": "A",
  "source": {
    "run_id": null,
    "created_at": null
  }
}
```

本例的未知 timestamp 必須在報告列 pending，附備份內 `params.json#fluxdep_fit` 的定位，不能把它標為已驗證的歷史時間。
若同一份已核對參數缺少單位或兩個允許來源互相衝突，把 `flux_unit` 寫成 `null`，保留數值，不換算。
報告逐項記錄兩個來源、原值或缺失，並列 pending。這份檔案不代表磁通座標已可安全使用。
`cur_A` 名稱、逐欄 unit、目前裝置、其他區塊或 meta_info 的單位都不能補作 predictor 的單位證據。
predictor 的單位與數值在同一個 JSON，不寫進 `general.ext.flux_unit` 或 qubit 的 local `flux_unit`。
保留完整原始 params 及 metadata 的 byte-identical 備份，不能只靠上述精簡 JSON 取代來源稽核。

### 其餘 params 與 meta_info 寫入 setup

編輯 `$NEW_R/setup.yaml`，使用[正式 component 欄位](../zcu_lab/components/)。
沒有正式欄位的區塊，如 dispersive、t1_curve_fit、plot_transitions，將整個原始區塊連同內含 metadata 放進
`components.<qubit>.ext.legacy_params.<區塊名>`。不能只挑 params 數字或刪掉未知子欄位。
另列 located pending，保留來源 JSON 的 byte-identical 備份。
project/name/schema_version 等識別資料也記入稽核清單，沒有正式對應的內容原樣保留於 legacy_params，
不冒充 entry UUID 或量測資訊。

再套用根層 meta_info。已有 meta 值優先，params 只補有正式對應的缺值；這不包含已獨立保存的 fluxdep_fit。
不一致時保留 meta，報告列兩個來源並 pending。predictor 不能作為 setup 的參數或單位後備來源。
以下是合成 setup 範例，假設根層 meta 明示 `q_f: 5000.0`，legacy_params 保存原始 dispersive 區塊。
UUID 與 created_at 應保留剛建立的實值。

```yaml
format: zcu.parameter-container
format_version: '2.0'
general:
  entry_id: '54cd2da2-e8cd-49d5-a414-4ec40eb7eb14'
  created_at: '2026-10-20T08:00:00Z'
  description: 'Manual migration of one legacy qubit'
components:
  Q0:
    kind: qubit/fluxonium
    freq: 5000.0
    ext:
      legacy_params:
        dispersive:
          params: {chi: 0.12}
          metadata: {comment: 'original analysis note'}
metadata:
  Q0.freq:
    source: 'meta_info.json#q_f'
    comment: 'Copied in MHz, no conversion'
```

meta_info 的主要對照如下。Q、R、J 是你已確認的實際 component 名稱，不是角色字串。
完整歷史 key 清單見 [storage_mapping.py](../migrations/storage_mapping.py)。未列出的 key 保全並 pending，不猜測用途。

| 舊 key | 新 logical path |
| --- | --- |
| `q_f`、`qf_w`、`t1`、`t2r`、`t2e` | `Q.freq`、`Q.kappa`、`Q.t1`、`Q.t2r`、`Q.t2e` |
| `r_f`、`rf_w` | `R.freq`、`R.kappa` |
| `qub_0_1_ch`、`lo_flux_ch`、`res_ch`、`ro_ch` | `Q.wiring.ch`、`Q.wiring.lo_flux_ch`、`R.wiring.ch`、`R.wiring.ro_ch` |
| `flux_unit`、`cur_A` | `general.ext.flux_unit`、`general.ext.flux_value`，數字須同一 meta JSON 有明示 flux_unit |
| `flx_half`、`flx_int`、`flx_period` | `Q.flux_half`、`Q.flux_int`、`Q.flux_period`，同上 |
| `local_flux_unit`、`jpa_flux_unit`、`best_jpa_flux` | `Q.flux_unit`、`J.flux_unit`、`J.flux`，J 數字須有 jpa_flux_unit |
| `t1err`、`t2r_err`、`t2e_err` | `metadata[Q.t1/Q.t2r/Q.t2e].stderr`，保留工作單位 |
| `pi_gain`、`pi_len`、`pi2_gain`、`pi2_len` | module/waveform 資料，不直接塞 qubit scalar；依下一節的正式 Spec 處理 |

頻率保留 MHz，時間保留 us，能量保留 GHz，metadata 不換算。
meta_info 的新值寫在 `components` 或表列的 `general.ext`，metadata 的 key 不加 `components.` 前綴。
這些正式 component 欄位的 mapping 不因 predictor 獨立保存而改名或移除。
若缺 R/J 的實體對應，相關欄位 pending，不為了套表新增虛構元件或通道。
完成時逐欄比對 params 與根 meta；setup 中每個採用值都有來源，pending 不混入可用數值。

## 5. 複製完整 YAML 成 point，再套 point meta

先完成 setup，再逐個建立 `exps/<sub>` 對應的 point。不要逐區塊挑 components 或 metadata。
整份 copy 才能保存 `general.ext` 與其他合法內容，避免 meta_info 的 flux 有數字卻沒單位。
predictor 留在 entry 的 `predictors/`，不複製到 point，也不從 JSON 把 fit 參數補進 point.yaml。
本節用直接檔案操作，不依賴 checkout 的 `new_point` 是否已更新繼承行為。

```bash
export LABEL=070716_0.500
export POINT="$NEW_R/points/$LABEL"
test ! -e "$POINT"
mkdir -- "$POINT"
cp -- "$NEW_R/setup.yaml" "$POINT/point.yaml"
printf "format: zcu.module-library\nformat_version: '2.0'\n" > "$POINT/module_cfg.yaml"
date -u +%Y-%m-%dT%H:%M:%SZ
```

預期有完整 point.yaml 與 header-only library。用編輯器做以下修改：

- 為相容目前 main 的 [PointGeneral loader](../lib/zcu_tools/resources/entry/schema.py)，刪除 point 副本中的 `general.entry_id`。目前版本會拒絕此欄，這不是永久格式禁令。
  setup 的 UUID 必須保留，不另造 point UUID。
  owner 更新為整份來源複製並接受 optional seed UUID 後，可保留來源 UUID。下方省略此欄的最小格式仍可讀，不依賴該更新。
- `general.created_at` 改為剛才印出的建立時間。這是新 point 的建立時間，不是歷史量測時間。
- description 未明確指定時保留來源描述；明確提供字串時使用該字串，空字串也有效。
- 其他全部內容保留，再套 `exps/<sub>/meta_info.json`。meta 提供值優先；缺值才繼承 setup。

point 最小形狀如下，完整文件仍須包含來源所有 components 與 metadata：

```yaml
format: zcu.parameter-container
format_version: '2.0'
general:
  created_at: '2026-10-20T08:05:00Z'
  description: 'Manual migration of one legacy qubit'
components:
  Q0:
    kind: qubit/fluxonium
    freq: 5000.0
    ext:
      legacy_params:
        dispersive:
          params: {chi: 0.12}
          metadata: {comment: 'original analysis note'}
metadata:
  Q0.freq: {source: 'meta_info.json#q_f'}
```

如果 point meta 的 global unit 與 setup 中根 meta 的單位不同，移除 point 副本中僅從根 meta 繼承的
flux_half/int/period 及其 metadata，再寫入有該 point meta 單位證據的數值。
不要留下「根 meta 的 A 數字配上 point 的 V」。缺少 point 數字就保持缺值，報告列兩來源 pending，不換算。
相同單位但數值不同也列衝突，保留 point meta。這只處理 meta_info 的正式 component 欄位，不改 entry predictor。

從另一個 point clone 時，整份來源改為該 point.yaml，不帶入 setup 的較新值。
library 的複製與來源處理另依 context owner；不要直接複製帶 ledger event 引用的 library 到另一 entry。
本次由舊 exps 建 point 時，依各自的舊 module_cfg 轉換，不靠 root library 推測。

完成判斷是每個舊 point 都有唯一對應的新 point、完整 metadata 與正確 global unit。
已存在 point 不執行上述建立命令，依部分接手逐欄補缺，不覆寫整份文件。
`entry.list_points()` 只列同時有 point.yaml 與 module_cfg.yaml 的目錄，因此也要用
`find "$NEW_R/points" -mindepth 1 -maxdepth 2 -print` 查未完成骨架，不能把漏列當成沒有資料。

### module_cfg 轉正式 Spec

舊文件的 `modules` 與 `waveforms` mappings 不是新 library。
逐筆建立「來源 family、舊 key、新 canonical key、引用位置」對照，再轉到
`components.<component>.{pulses,readout,reset,waveforms}.<name>`。
例如已確認舊 `pi_amp` 是 Q0 的 pulse，可對應 `Q0.pulses.pi_amp`，並在 point 的
`components.Q0.module.x180` 寫該 key。不能只填 module slot 卻沒有有效 Spec。

下面是合成的合法新格式示意。通道、頻率、gain 與波形長度都要由舊檔證明，不使用範例值補缺。

```yaml
format: zcu.module-library
format_version: '2.0'
components:
  Q0:
    pulses:
      pi_amp:
        type: pulse
        ch: 0
        nqz: 1
        freq: 5000.0
        gain: 0.1
        waveform: Q0.waveforms.drive
    waveforms:
      drive: {style: const, length: 0.1}
provenance: {}
```

操作順序與完成條件：

1. 先用 [cfg_model](../lib/zcu_tools/cfg_model/README.md) 的公開 `module_spec_from_raw(raw)` 或 `waveform_spec_from_raw(raw)` 驗證每筆候選。用 `remap_module_references`／`remap_waveform_references` 和明示 mapping 改 child keys。不 evaluate expression，也不猜必要欄位。
2. 通過後用 `entry.module_library(label).edit()` 的 `set_module(key, spec)`／`set_waveform(key, spec)` 提交，範例如下。普通手動寫入由 owner 產生 manual provenance，不虛構 accepted event 或 run_id。最小文件可省 provenance；透過 owner 編輯後，以實際產出的來源表為準。
3. readout 要符合正式 `readout/pulse` 平鋪 Spec，generator 是 `gen_ch`，ADC 是 `ro_ch`，只保留一個 freq。舊 nested `pulse_cfg`/`ro_cfg` 不能只改 header。若不能證明如何重組，該筆 pending。
4. 以 `library.snapshot().module_spec(key)`／`waveform_spec(key)` 讀回核對，並檢查每個 child reference。需要 asset 才能 materialize 時，明確提供已驗證的 asset capability。Spec 可讀不等於缺失 asset 已補齊。

以 `uv run --directory "$REPO" --no-sync -- python` 開啟離線 Python 工作階段。
依第 8 節註冊 models 並 open entry 後，可依候選內容提交。例如下面只示範已確認的 const waveform，`raw_waveform` 必須取自已核對來源：

```python
from zcu_tools.cfg_model import waveform_spec_from_raw
spec = waveform_spec_from_raw(raw_waveform)
with entry.module_library(label).edit() as draft:
    draft.set_waveform("Q0.waveforms.drive", spec)
print(entry.module_library(label).snapshot().waveform_spec("Q0.waveforms.drive"))
```

預期印出與來源相同的 Spec；Q0 與 key 都換成已確認的對應。
未知 family、未支援 expression、缺 mapping、缺欄位或引用不能轉換者，留備份並逐筆 pending。
空 library 合法，但不能宣稱舊 library 已全部轉換。
注意 library 的一般讀取可能備份並清理壞條目，先在工作副本檢查候選，不用開啟真檔來試錯。

## 6. 複製 figures 與需保全的檔案

所有檔案複製都先查目的地。不存在才複製；已存在且 bytes 相同視為完成；不同就停下記衝突。
以下 helper 只處理普通檔，不覆寫既有檔案，也不刪來源：

```bash
copy_checked() {
  local src=$1 dst=$2
  test -f "$src" && test ! -L "$src"
  test ! -L "$dst"
  mkdir -p -- "$(dirname "$dst")"
  if test -e "$dst"; then
    cmp -- "$src" "$dst"
  else
    cp --no-clobber -- "$src" "$dst"
    cmp -- "$src" "$dst"
  fi
  sha256sum -- "$src" "$dst"
}
shopt -s nullglob
for point_src in "$BR"/exps/*; do
  test -d "$point_src" || continue
  label=$(basename "$point_src")
  test -f "$NEW_R/points/$label/point.yaml"
  for png in "$point_src"/image/*.png; do
    copy_checked "$png" "$NEW_R/points/$label/figures/$(basename "$png")"
  done
done
copy_checked "$BR/params.json" "$NEW_R/params.json"
# 對盤點中實際存在且需保全的每個檔呼叫，例如：
copy_checked "$BR/samples.csv" "$NEW_D/migration-preserved/result/samples.csv"
```

預期每一對 SHA256 相同，`cmp` exit 0。存在非檔案目的地或不同 bytes 時停止，不自動改名。
helper 不是跨檔交易，中斷可能留下半份目的檔；先記為未完成 publication，從備份核對並取得處理授權後再補。
不要把半份檔案認作永久 pending 然後清理來源。

對 autofluxdep_runs、exps 外 image 與未知檔，逐檔以相對路徑呼叫 helper 到 preservation 位置。
從 `$BR/arb_waveforms/` 與 `$BD/arb_waveforms/` 各自存在的檔案逐檔複製到 `$NEW_D/arb_waveforms/`，保留各自 arb_waveforms 下的相對 asset key 與目錄關係。
兩來源若落在同一目的，bytes 相同才視為同一副本；不同就停下，報告列兩來源與 hash，不覆寫。
預期每個 preservation 副本 hash 都等於備份，報告記明它不是新格式資料。
舊 samples 要成新 JSONL，另需[樣本 owner](../lib/zcu_tools/resources/sample_table/README.md) 要求的列、座標與來源證據。
沒有證據就保全 pending，不把 CSV 改副檔名成 JSONL，也不把舊 autofluxdep manifest 放進 workflows 假裝新 run。

## 7. Labber 有證據才產 native

Labber 與 native 都可能是 HDF5，但格式不同。把 `.hdf5` 改名為 `data.h5` 不會得到 native。
不要從路徑日期、mtime、檔名、目前 setup 或目前裝置猜量測時間、cfg、point、run_id 或 ledger。

每個檔案先核對下表。experiment tag+cfg_type、完整歷史 cfg，以及 variable/axes/signal 的名稱、單位、dtype、shape 都必須齊全。
缺少這些或其他必要 mapping 時，停止該檔轉換，保全並列永久 pending，不推論、不拿目前值補缺。
只有開始時間與 run_id 依下列規則處理，不因這兩項未知就擋住 native 產出。

- 缺可證實開始時間時，仍寫 native 2.1，使用 `started_at=None`、`completion="unknown"`、`finished_at=None`。這不是量測完成的證據。
- 缺可信 run_id 時，用公開的 `resources.entry.new_run_id()` 分配，報告標「遷移賦值」。ID 裡的 UTC 時戳是本次分配時間，不是歷史量測時間，不能拿來填 started_at。
- 可信既有 run_id 與可證實歷史時間保留原證據。已知開始時間仍寫 UTC ISO；完成狀態只採證據支持的 complete/partial/stopped，未知用 unknown，不把可讀資料等同 complete。

本節是手動流程，使用目前已公開的 native 2.1 API，不表示 `migrate_storage` 已支援這些未知歷史值。

| 證據 | 要填的正式資料 |
| --- | --- |
| 來源相對路徑與 SHA256 | 報告中的原檔身分，必須與備份一致。 |
| experiment tag、cfg_type、完整 schema_version | `(tag, cfg_type)` 配對到明確 cfg 與 disk schema，不能只看 tag。 |
| 完整歷史 cfg JSON | `CfgSnapshot(values, cfg_type, schema_version)`，不拿目前 defaults 補值。 |
| variable、axes、signal 的名稱、SI/離散單位、dtype、shape | `VariableSchema` 與 `ExperimentPayload`，inner-first axes，array shape 是反向 axes 長度。 |
| started_at、completion，及可得的 finished_at 證據 | `RunMetadata` 保留可證實的 UTC ISO 時間與完成狀態。缺開始時間時用 None/unknown/None；已知開始但完成狀態未知時 completion 用 unknown。finished_at 不明用 None，不以檔案時間代替。 |
| 當時 entry 名、point、description、roles，以及此次 entry UUID 的明示對應 | `RunSnapshot`，point 可有證據地為 None；新 UUID 的對應是遷移指定，不聲稱舊系統有此 UUID。 |
| 歷史軟體/主機/SoC 資訊 | `SoftwareProvenance`，未知只用型別允許的 None/空 mapping，device_snapshot 未記錄時用 None，不寫空表冒充捕捉。 |
| 可信 run_id，或本次遷移的 ID 分配紀錄 | 保留可信既有 ID。缺可信 ID 時用 `new_run_id()` 分配，先保存來源/hash 與 ID 的對照，報告標「遷移賦值」，不冒充歷史 ID 或時間。接手沿用已固定 ID。 |

正式型別與欄位見 [datafile owner](../lib/zcu_tools/datafile/README.md) 與
[native models](../lib/zcu_tools/datafile/native_models.py)。ID 分配使用 [entry owner](../lib/zcu_tools/resources/entry/README.md#輸出位置)
的 `new_run_id(*, at=None)`，不自行拼接 ID，也不從 run_id 解析歷史時間。
以下示範必要 mapping 已齊全，但開始時間與可信 run_id 不可得的情況。
這是合成範例，所有數字、UUID、名稱與路徑都不可直接用在真檔。
假設虛構實驗 `synthetic_demo` 的 `DemoCfg` 完整 cfg 只有 reps，資料是單一 frequency 軸的複數 response；它不是 repo 已註冊的實驗。
真檔必須換成已確認的 `(tag, cfg_type, schema_version)`、完整歷史 cfg JSON 與正式 disk schema，不可只換檔名或套用這份 reps。
None 與空 provenance mapping 示範「證據確實不可得」，不是略過查證。其他必要證據缺失仍 pending。
先走部分接手盤點，確認沒有該來源的有效 native 或已固定 ID，再執行新分配。
範例將對照保存在 WORK 的獨立 JSON；它是手動紀錄，不是工具 resume manifest。

```python
import hashlib
import json
import os
from pathlib import Path
from uuid import UUID
import numpy as np
from migrations.legacy_labber import load_legacy_labber_payload
from zcu_tools.datafile import (
    AxisSchema, CfgSnapshot, DataVariable, RunMetadata, RunSnapshot,
    SoftwareProvenance, VariableSchema,
    load_run_data, save_run_data, validate_experiment_payload,
)
from zcu_tools.resources.entry import new_run_id

source = Path(os.environ["BD"]) / "synthetic-only.hdf5"
with source.open("rb") as stream:
    source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
# Compare source_sha256 with the verified backup inventory before continuing.
identity_path = Path(os.environ["WORK"]) / "synthetic-only-native-identity.json"
if identity_path.exists():
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (identity["source"] != source.relative_to(Path(os.environ["BD"])).as_posix()
            or identity["source_sha256"] != source_sha256):
        raise ValueError("Saved identity does not match the verified source")
else:
    # This example has no trusted historical ID or previously assigned ID.
    identity = {
        "source": source.relative_to(Path(os.environ["BD"])).as_posix(),
        "source_sha256": source_sha256,
        "run_id": new_run_id(),
        "run_id_origin": "遷移賦值",
    }
    with identity_path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, ensure_ascii=False, indent=2)
print(identity_path, identity["run_id"], identity["run_id_origin"])

schema = (VariableSchema(
    variable=DataVariable("response"),
    axes=(AxisSchema(name="frequency", unit="Hz", dtype=np.dtype("float64")),),
    signal_name="signal", signal_unit="V", signal_dtype=np.dtype("complex128"),
),)
cfg = CfgSnapshot(
    values={"reps": 10}, cfg_type="DemoCfg", schema_version="1.0",
)
metadata = RunMetadata(
    run_id=identity["run_id"], experiment="synthetic_demo",
    started_at=None, finished_at=None,
    completion="unknown", labber_path=None,
    snapshot=RunSnapshot(
        entry_id=UUID("54cd2da2-e8cd-49d5-a414-4ec40eb7eb14"),
        entry_name="migrated-Q0", point="070716_0.500",
        description="Synthetic example only", roles={"qubit": "Q0"},
    ),
    provenance=SoftwareProvenance(
        software_versions={}, git_commit=None, git_dirty=None,
        qick_version=None, soc_fingerprint=None, hostname=None, simulation_id=None,
    ),
    device_snapshot=None,
)
destination = Path(os.environ["NEW_D"]) / "runs" / metadata.run_id / "data.h5"
payload = load_legacy_labber_payload(source, schema=schema)
validate_experiment_payload(payload, schema=schema)
destination.parent.mkdir(parents=True, exist_ok=True)
save_run_data(destination, payload, metadata, cfg=cfg)  # 不使用 replace=True
stored = load_run_data(destination)
validate_experiment_payload(stored.payload, schema=schema)
```

預期先得到一份固定的來源/hash→run_id 對照，再得到 `runs/<該 run_id>/data.h5`。
把對照中的 run_id、`run_id_origin="遷移賦值"`、備份內完整路徑及 native path 記入 manual-report.json 的該來源列。
另記 `started_at=null`、`completion="unknown"`、`finished_at=null` 與「缺可證實開始時間」，不要把這些值報成歷史時間已驗證。
只缺上述歷史值不列為無法產 native 的 pending；其他必要證據不足仍列 pending。
「遷移賦值」只在報告與手動對照中說明，不加進 native metadata 的正式欄位。

中斷後保留並讀取同一份對照，不刪檔重配 ID。對照損壞、來源/hash 不符或與既有 native 衝突時停止處理。
若先前工具或人工已固定 ID，先核對其來源/hash 並沿用，不能因本例的對照檔不存在就再分配。
有可信歷史 ID 時，改以該 ID 建立對照，origin 記其證據來源，不呼叫 new_run_id。
目的 native 已存在時不要重跑寫入段，先以 reader 核對原 ID 與資料，只補缺項，不用 replace=True。

schema 只宣告名稱、單位與 dtype，axes 依 inner-first 排列；軸值、signal arrays、shape、timestamps 與 Labber metadata 由來源 payload 讀入並驗證，不在此補造。
此例單軸 signal shape 必須為 `(該軸長度,)`；多軸 shape 為反向 axes 長度。
HDF5 階層與序列化交給 [datafile owner](../lib/zcu_tools/datafile/README.md) 的 writer，不手填 dataset。
此段是 owner API 的操作順序，不是能替任意 Labber 猜出 schema 的通用轉換器。
先依該實驗的正式 cfg model 驗證歷史 cfg；寫完還要由具體 experiment 的 `load_run` 讀回，
逐欄比對 cfg、tag、時間、snapshot、axes、signal、timestamps 與原 payload。
本例的 generic metadata 讀回必須仍是 None/unknown/None，run_id 必須與保存的對照相同。
需要核對 typed metadata 時，用 `experiment.native_persistence.load_run_with_metadata(path, spec=...)`，不可把 unknown 改成 complete。
預期 generic 與 typed reader 都成功，內容相同。complete-only 分析入口可能拒絕 unknown，不能用它判定 native 無效。
讀回失敗時保留來源，記未完成 publication，不搬走唯一副本。
對未知 pair、未支援 layout、缺歷史 snapshot 的檔案，此時無法手動安全補成 native，應明示限制而不是補猜測。

只有 native 讀回通過後，才用 `copy_checked` 將原 Labber 存入
`$NEW_D/Labber/<原相對路徑>`，保留原檔名與 bytes，不套現行新量測檔名規則。
核對兩份 hash，把對應 native path、Labber path 與原檔 hash 記入報告。
無法證明原始日期結構時，改存 preservation，不偽裝 SaveLayout 產物。
遷移 native 的 `labber_path` 可明示 None，原檔關係由報告保存。
本流程不手造 acquired/saved/accepted ledger 事件。沒有事件不代表檔案失效，也不能聲稱 GUI ledger 清單會自動出現它。

## 將他機已寫入的 results 搬到 result

已在他機寫入 `results/<name>/` 的完整或部分 entry，只需搬到 `result/<name>/`，不需重遷。
這是實體目錄搬移，不是 native 格式轉換。不改 entry UUID、point label、metadata 或任何檔案內容，
也不搬動或修改 `Database/<name>/`。未完成的骨架、隱藏檔、暫存檔、報告與未知檔都一起保全。
搬完不代表遷移完成，原有 pending 與清理限制仍適用。

先停止 GUI、Notebook、同步程式與其他 writers，取得本次備份與搬移授權。
以下範例在同一個 Bash shell 依序操作，使用 GNU 工具。換成已確認的絕對路徑與單一路徑段 NAME，
不含 `/`、`.` 或 `..` 路徑段。WORK 使用 repo 外尚不存在的獨立目錄。

```bash
set -euo pipefail
export REPO=/absolute/path/to/Qubit-measure-gui
export NAME=migrated-Q0
export WORK=/absolute/path/outside/repo/root-move-review
export FROM_ROOT="$REPO/results" TO_ROOT="$REPO/result"
export FROM="$FROM_ROOT/$NAME" TO="$TO_ROOT/$NAME"

# Reject symlinks in both roots, entries, WORK and all parent directories.
no_symlink_parents() {
  local path=$1
  while test "$path" != /; do
    test ! -L "$path" || return 1
    path=$(dirname -- "$path")
  done
}
for path in "$FROM" "$TO" "$WORK"; do
  no_symlink_parents "$path"
done
realpath -m -- "$FROM_ROOT" "$TO_ROOT" "$FROM" "$TO" "$WORK"
test -d "$FROM"
test ! -e "$TO"
test ! -L "$TO"
test ! -e "$WORK"
test ! -L "$WORK"
find "$FROM" ! -type d ! -type f -print
```

預期 realpath 列出已確認的絕對路徑，所有 test exit 0，find 無輸出。
人工核對兩 root 不相同、不互相包含，來源與目的不重疊。WORK 不在 repo 內，與兩 root 也不互相包含。
若發現 symlink 或特殊檔，停止並確認保全方式，不跟隨連結搬移。
**目的已存在就不執行下方命令。** 即使看似空目錄，也先保存兩端完整副本與清單，
按[部分遷移接手](#從部分遷移的狀態手動接手)逐項比對，只補缺項，不覆蓋或混合整棵目錄。
來源中若記錄了絕對路徑或工具續跑狀態，另記新舊位置供 owner 確認，不為了搬移改寫原內容。

確認以上條件後，先保存完整副本與相對路徑清單。inventory 包含空目錄、檔案類型與權限，
hash 清單包含所有普通檔，包括未完成內容。

```bash
hash_tree() {
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z |
    xargs -0 -r sha256sum)
}
inventory_tree() {
  (cd "$1" && find . -printf '%y %m %P\0' | LC_ALL=C sort -z)
}
mkdir -- "$WORK"
hash_tree "$FROM" > "$WORK/before.sha256"
inventory_tree "$FROM" > "$WORK/before.inventory"
cp -a -- "$FROM" "$WORK/entry-before-move"
hash_tree "$WORK/entry-before-move" > "$WORK/backup.sha256"
inventory_tree "$WORK/entry-before-move" > "$WORK/backup.inventory"
cmp "$WORK/before.sha256" "$WORK/backup.sha256"
cmp "$WORK/before.inventory" "$WORK/backup.inventory"
hash_tree "$FROM" > "$WORK/recheck.sha256"
inventory_tree "$FROM" > "$WORK/recheck.inventory"
cmp "$WORK/before.sha256" "$WORK/recheck.sha256"
cmp "$WORK/before.inventory" "$WORK/recheck.inventory"

# Recheck the destination. Keep writers stopped during verification and move.
no_symlink_parents "$TO"
mkdir -p -- "$TO_ROOT"
test ! -e "$TO"
test ! -L "$TO"
mv -T --no-clobber -- "$FROM" "$TO"
test ! -e "$FROM"
test ! -L "$FROM"
hash_tree "$TO" > "$WORK/after.sha256"
inventory_tree "$TO" > "$WORK/after.inventory"
cmp "$WORK/before.sha256" "$WORK/after.sha256"
cmp "$WORK/before.inventory" "$WORK/after.inventory"
printf 'Root move verified: %s\n' "$TO"
```

預期所有 cmp 無輸出且 exit 0，最後印出 `Root move verified: .../result/<name>`。
完成判斷是來源 entry 已不存在，目的與搬前的完整 hash、inventory 一致，完整副本仍保留在 WORK。
記錄搬前後位置、驗證結果與日期，再用明確的 result/Database 根目錄重新開啟 entry，核對原 UUID 與 points。
不要沿用搬移前已開啟、仍綁舊路徑的物件。原有工具能否 resume 由 owner 確認，不改 manifest 強迫通過。
任何中斷或核對失敗都停止，不刪來源殘留、目的半份內容或備份，也不自動重跑 mv。
先保全現況，再走部分接手。不要刪整棵 results root，裡面可能還有其他 entry。

## 從部分遷移的狀態手動接手

不重新 create entry，不靠應急工具的 manifest 判定成功，也不手造正式工具 manifest 冒充 resume。
先做第 1 至 3 節，保存既有新 entry 的完整副本與逐檔 hash 到 WORK，再開啟修改。
若舊檔已被移除，先完成完整備份重建；只找到新副本不代表知道舊來源完整範圍。

用備份的 source inventory 作左欄，現有新 entry 作右欄，逐項填報告：

| 實際狀態 | 手動補法 | 預期輸出與完成判斷 |
| --- | --- | --- |
| 缺某些 points | 先補 setup 的缺失 params，再依第 5 節為缺項完整 copy YAML，套其 meta/library。 | 每個舊 label 唯一對應 point；已存在的 point 不重建。 |
| point 已有新修改 | 逐欄比較 params、根 meta、point meta 與當前值，只補缺值。 | 保留有效新值與原 metadata；不同值列兩來源 pending，需使用者決定才覆寫。 |
| PNG 在 `migration-preserved/result/exps/<sub>/image`，未掛 point | 對照原始相對路徑與備份 hash。建好/驗證 point 後，從已驗證副本用 copy_checked 複製到該 point/figures。 | 原檔名與 hash 相同；preserved 多一份不算衝突，不急著刪它。 |
| params.json 只 copy，缺 `predictors/<component名>.json` | 確定唯一 qubit，依第 4 節將舊 EJ/EC/EL fit 寫成 fluxonium，bias=0.0，不寫 flux_int。核對兩個允許的 unit 來源與 timestamp。 | zcu.predictor v1 的必填 key 齊全，run_id=null；未知參數可省略/null，未知時間為 null 並 pending。未定義的 transmon/明示 model fit 保全並 pending，不造 mapping。component 沒有 fit 不再是缺項。 |
| predictor 已存在或只寫了一半 | 對照備份逐欄核對 format/version、model、params、flux_unit、source；保存現檔，按契約補缺。 | 已有不同值保留並列衝突，不覆寫；未知參數省略/null，unit、run_id、created_at 不可省略。unit 缺失/衝突或時間無證據須為 null 且 pending。只改 header 不算完成。 |
| fit 曾被寫進 setup/point component | 保存現況與來源，按第 4 節另寫 entry predictor。逐欄確認舊 fit 的誤寫範圍，再修正 setup 及每個既有 point。 | 只移除能證實僅由 fit 誤寫的欄位與 metadata；meta_info 自身的正式欄位和有效新值保留。來源不明列 pending，先核對，不整份覆寫。 |
| 其餘 params 區塊未保存 | 依第 4 節補入對應 component.ext.legacy_params，保留完整原文及 metadata。 | setup 與每個既有 point 分別核對，不能重新 copy 蓋掉新值；無正式欄位者仍列 located pending。 |
| point 有 meta_info 的 flux 數字卻缺 unit | 查根 meta、該 point meta 與目前值的來源，依原有 meta 單位規則補缺。 | 無/衝突則移除僅遷移帶入且無證據的數字，保留原文與 pending；不借用 predictor 或 params 的單位。 |
| library 只換了 header 或有 dangling slot | 先保存現況，依正式 Spec 與 mapping 修補缺項，不覆寫有效 Spec。 | Spec、child keys、slot 與 asset 都有核對結果；不能轉者逐筆 pending。 |
| Labber 已搬到新處 | 以原始盤點/hash 對照現檔，確認備份包含原 bytes。另用正常 reader 查 native 是否真的存在且有效。 | 「已搬」與「native 已轉」分開記；只有改檔名不算完成。 |
| 已有有效 native 或已固定 run_id | 以來源/hash 對照、generic/typed reader 核對既有資料與 ID。只補缺少的副本或報告。 | 不因新規則覆寫有效 native，不重分配已固定 ID，不改可信歷史時間或已有 None/unknown。 |
| Labber 未轉 native，必要 evidence 齊全，只有開始時間或可信 ID 缺失 | 依第 7 節以 None/unknown/None 寫未知時間，缺 ID 用 new_run_id，先保存對照再產 native。generic/typed 讀回後 copy/hash 原 Labber。 | 未知時間與「遷移賦值」記入報告，不冒充歷史；重做時沿用固定 ID。開始時間已知則保留證據，不改為 None。 |
| Labber 未轉 native，有完整歷史 evidence | 保留可信 ID 與歷史時間，依第 7 節產 native、generic/typed 讀回，再 copy/hash 原 Labber。 | 新 run 與原檔的對照清楚，舊來源保留到最後統一清理。 |
| Labber 已搬或未搬，但其他必要 evidence 不足 | tag+cfg_type、完整歷史 cfg、variable/axes/signal 名稱、單位、dtype、shape 或必要 mapping 缺失時仍永久 pending。不猜 cfg/時間，不造 ledger。 | 完整備份及報告列明路徑、hash、缺項與理由。僅缺開始時間或可信 ID 不屬於此列。 |
| 來源和目的都在但 hash 不同 | 保存兩者，不挑較新 mtime，也不直接覆寫。 | 記為衝突或未完成 publication，使用者決定前不得清理。 |

每補一項就重新讀回或 cmp，更新該項狀態。現有 metadata 若與舊資料不同也保留並列衝突，不能靜默 deep merge。
手動接手可能改動正式工具原先追蹤的檔案，因此之後不能保證可用原工具 resume。
若要切回工具，先由工具 owner 確認接管方式，不能刪 manifest 或改 revision 強迫通過。

## 8. 最後核對與安全清理

先讀回完整 entry。以下只驗證參數，不控制硬體；在 fresh process 使用明確註冊的 component models。

```bash
uv run --directory "$REPO" --no-sync -- python - <<'PY'
import os
from zcu_lab.components import register_all
from zcu_tools.resources.entry import ResultEntry, component_registry
register_all(component_registry)
entry = ResultEntry.open(
    os.environ["NAME"], result_root=os.environ["RESULTS"],
    database_root=os.environ["DATABASE"],
)
print("entry", entry.entry_id)
for label in entry.list_points():
    entry.use_point(label)
    print("point", label)
PY
```

預期列出原 entry UUID 與所有 points，exit 0。接著逐項審閱，不用程式載入成功代替內容核對：

- 每份 fluxdep_fit 已對應 entry 的 `predictors/<component名>.json`，或在備份定位並列永久 pending。逐欄核對嚴格整數 version、模型、模型與 flux 參數、unit、source，不加未知 key。entry loader 成功不代表已驗證 predictor。
- predictor 的未知參數省略或填 null。unit 缺失/衝突、timestamp 無證據時，對應必填欄位為 null 且 pending，報告記錄備份內原文位置。
- 其餘 params 已完整保存在正確 component.ext.legacy_params 並列 located pending。meta_info 仍寫正式 component/general 欄位。metadata、global/local unit、衝突處理均有記錄，fit 不混進 setup/point。
- 每張 exps PNG 已掛正確 point，原名與 hash 相同。沒有「等 point」的暫待圖。
- 每筆可轉 module/native 已由相應 owner 讀回，未完成、半份檔案與暫存 publication 都已處理。
- 缺開始時間的 native 2.1 讀回仍是 None/unknown/None。遷移分配的 ID 與固定對照一致，報告標「遷移賦值」，未把 ID 時戳當 started_at。可信既有 ID 與歷史時間未改寫。
- samples、autofluxdep、未知檔及無法轉 Labber 已保全。永久 pending 逐筆以 BACKUP 內路徑與 hash 定位，原因可追溯。
- parameters 與 data 兩部分均完成核對，報告的 unfinished_publications 與 temporary_pending 都空。只剩真正無法轉換的永久 pending 才可進清理。

再次執行第 2 節的備份 hash 驗證及完整清單比較，預期全部 exit 0。
舊來源仍在時也重建其清單比對基線；若出現新檔或變更，停止清理並重新盤點，不能刪掉遷移期間新寫入的資料。
保存新 entry 的最終 hash 清單與 manual-report.json 到 WORK，並各留一份報告在
`$NEW_R/records/manual-migration-report.json` 與 BACKUP 根目錄。
不要覆寫既有報告，採有日期的獨立檔名。報告必須能在舊來源刪除後定位永久 pending。

只有使用者確認以上結果且明確授權本次刪除後，才執行下列操作。
先重新核對 realpath，確保兩條都是本次舊 `<chip>/<qub>`，不是 result/Database 根、BACKUP 或新 entry。

```bash
printf 'Delete only these old sources:\n%s\n%s\n' "$OLD_R" "$OLD_D"
realpath -- "$OLD_R" "$OLD_D" "$BACKUP" "$NEW_R" "$NEW_D"
# 人工核對路徑、備份驗證結果、報告與刪除授權後，另行執行：
rm -ri -- "$OLD_R" "$OLD_D"
test ! -e "$OLD_R" && test ! -e "$OLD_D"
```

預期只刪這次的兩個舊來源，最後檢查 exit 0。不要刪其他 qubit、父日期目錄或整棵 Database。
保留 repo 外完整 BACKUP、hash 清單、報告與失敗紀錄。永久 pending 可以只存在備份，但不能只寫「未遷移」而沒有檔案定位。
刪後重讀新 entry、確認 figures/native 還在，將清理日期、範圍與結果補入報告。

### TestChip/Q0 的核對基準

repo 範例的盤點為 5 個 exps、14 張 PNG，label 是 `062823`、`070216`、`070716_0.500`、
`070800_0.500`、`072523`。params.json 只有 schema_version、project、name，沒有 fluxdep_fit，
所以不能從這份範例憑空產出 predictor，或把合成範例的 EJ/EC/EL 當成實值。

舊流程的唯讀 dry-run 回報 exit 0、key_mappings 0、failures 0、pending 28，
其中 association 18、unconverted file 10。該次使用獨立空 Database 作基準，沒有驗證真實 Labber。
這顯示舊流程漏掉 exps 與 point 圖的限制，不是全部成功報告，也不是本指南已遷移資料的證據。
手動完成後應有 **5 個對應 points、14 張 point figures**，逐張 hash 一致。
samples/autofluxdep 等仍依證據保全或 pending，Labber 的完成數另以該機實際盤點為準。
