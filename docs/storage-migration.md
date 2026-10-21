# 舊資料遷移指南

支援的檔案與格式就轉換，不支援的就原樣保留。欄位也一樣，未知欄位以原名原值留在新容器，報告區分已轉換與原樣保留。Pending 僅用於支援項目缺必要證據而無法轉換，原文仍須可見。非法輸入與安全衝突不能當成成功。

本指南把一個舊 qubit 的資料，也就是 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`，整理成一個新格式 entry。
正常遷移使用下列工具指令，中斷或完成後都可直接重跑。第 1 到第 9 步另提供不用遷移腳本的手動路線。

## 遷移完成後的樣子

```text
<父目錄>/
├── Qubit-measure-gui/                    ← 原 repo：只剩新格式
│   ├── result/<name>/
│   │   ├── setup.yaml
│   │   ├── module_cfg.yaml               ← 舊根層 module_cfg 轉成的 setup library
│   │   ├── predictors/<Q>.json           ← 舊 fluxdep_fit
│   │   ├── dispersive/<Q>.json           ← 舊 dispersive ＋ plot_transitions
│   │   ├── t1_curve/<Q>.json             ← 舊 t1_curve_fit
│   │   ├── samples/<Q>.jsonl             ← 舊 samples.csv
│   │   ├── figures/<原相對路徑>           ← exps 以外的舊圖
│   │   ├── <原相對路徑>                   ← 不支援的舊檔，原樣放在對應位置
│   │   ├── points/<label>/{point.yaml,module_cfg.yaml,figures/}
│   │   └── records/manual-migration-report-<date>.json
│   └── Database/<name>/
│       ├── runs/<experiment>_<run_id>/data.h5  ← 完整 run_id，Labber 轉成的 native
│       ├── Labber/<原相對路徑>            ← 所有原 Labber 檔（含未轉 native 的），bytes 不變
│       ├── arb_waveforms/**
│       └── <原相對路徑>                   ← 無法轉換的舊 Database 檔，原樣放在對應位置
└── Qubit-measure-gui-backup/             ← 平行備份根：舊格式，路徑與 repo 內相同
    ├── result/<chip>/<qub>/              ← 原樣複製
    ├── Database/<chip>/<qub>/            ← 原樣複製
    └── migration-records/<chip>/<qub>/   ← hash 清單、工作紀錄、報告
```

Entry 本身只由 `result/<name>/` 構成。Database 是輸出位置，首次發布資料時才建立，不是開啟 entry 的前提。上圖的 Database 內容表示遷移已發布資料，不表示初始化會建立空 Database。

三條規則決定整個流程：

1. **平行備份。** 舊來源整份複製到 repo 旁的 `<repo 目錄名>-backup/`，保留 repo 內的相對路徑，逐檔核對 hash。每次遷移都累積到同一個備份根。
2. **新格式寫回原 repo。** 新 entry 直接寫在同一個 `result/`、`Database/` 根下，與舊來源不重疊。
3. **最後刪舊來源。** 新 entry 全部核對完、使用者授權後，刪掉 repo 內本次的 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`。repo 從此只剩新格式；舊格式只存在平行備份裡。

## 選路線

| 現況 | 走法 |
| --- | --- |
| 舊資料還沒動過 | 先用[工具路線](#工具路線與直接重跑)做 dry-run；要手動操作時依序做第 1 到第 9 步。 |
| 本工具已開始或完成同一份遷移 | 以原來源、entry 名稱與宣告直接重跑，不需先手動補檔或加 `--resume`。 |
| 工具回報差異 | 保存兩邊內容，按來源／目的與比較摘要逐項人工確認，不刪檔強迫重跑。 |
| 已有部分新 entry（應急工具或人工只做一半） | 做第 1 到第 3 步，再到[部分接手](#部分接手)。 |
| 他機把新 entry 寫在 `results/<name>/`（多一個 s） | 只需[搬到 result](#把-results-搬到-result)，不重做遷移。 |

## 工具路線與直接重跑

停止來源目錄的其他 writer。確認本次讀寫範圍，以及完成後刪除兩棵舊來源的授權。未取得刪除授權時，只執行 dry-run。以下路徑、kind、欄名與單位都須由操作者確認，不從檔名推論。

```bash
uv run --directory <checkout> --no-sync -- python -m migrations.migrate_storage \
  --repo-root <checkout> --backup-parent <external-parent> \
  --result-root <checkout>/result --database-root <checkout>/Database \
  --entry-result-root <checkout>/result \
  --source-chip <chip> --source-qubit <qubit> --name <new-entry> \
  --part all --qubit-kind qubit/fluxonium \
  --assume-flux-unit mA --assume-jpa-flux-unit V \
  --sample-value-column "calibrated mA" --sample-source-unit mA \
  --dry-run
```

核對 stdout 的 JSON report、pending 位置、備份與目的路徑。Dry-run 不建備份或目的、不寫報告，也不刪來源。確認後移除 `--dry-run`，其餘參數保持相同。Native 需要的歷史 evidence 另用 `--run-evidence <file>` 提供，缺必要證據的 Labber 保留原檔並列 pending，不猜 cfg 或量測時間。

正式執行先完整備份並逐檔驗 SHA256，再發布新格式。初始化只在 Result root 的已記錄暫存名稱下建 entry，核對 staging 後 rename 到正式 Result 名稱。Database 不參與 staging 或 rename，首次 data publication 才建立所需輸出目錄。

兩部分與所有 publication 完成後，工具重驗備份與來源，才刪本次舊來源。自動報告在 `<checkout-name>-backup/migration-records/<chip>/<qubit>/migration-report-<UTC-date>.json`。完成後另存到新 entry 的 `records/`，同處的 `migration-state.json` 保存固定身分與比較依據。

中斷時保留 entry、工具暫存檔、manifest 與備份，直接重跑原指令。工具只補缺項，沿用 entry UUID、建立時間、run_id 與報告日期。`--resume` 仍可明填；parameters 與 data 分開執行時，第二次改 `--part` 即可。單位、CSV 欄名、kind、mapping 版本與已固定的歷史 evidence 不得改成另一份宣告。

完成後也用原指令重跑。工具先核對完整備份，再比較 setup、points、libraries、分析 JSON、JSONL、processed 正式與 companion，以及 figures、raw、Labber 和 native。YAML／JSON 用 owner typed 語意並保留 unknown raw，排版不算差異。Binary 與 JSONL 使用已驗 SHA256。同內容不重寫、不追加、不重新分配 ID，stdout report 的 `skipped` 逐項列出 `source`、`destination`、`comparison` 與 `summary`，成功 exit 0。

不同內容，包括合法的人工改值，會在 stderr 列出來源、目的、比較種類、預期與實際摘要，exit 2。工具不覆寫目的、manifest 或原報告。需要保留診斷時，由操作者把 stdout／stderr 另存到獨立檔案，不指定受檢檔案作為輸出。執行或 I/O 故障 exit 1，保留現有狀態後重跑。

舊本工具 manifest 缺文件 baseline 時會報「缺比較依據」並 exit 2，需人工比對或在新備份上重新遷移。

來源清理後即使來源不在，工具仍從已驗 backup／manifest 重驗目的。把相同 baseline 的來源放回時，工具先驗全部目的，再成功略過並保留放回的來源，不刪第二次。來源不同則報差異。這些重驗也可帶 `--dry-run`，仍然零寫入。

獨立 `python -m migrations` 的 CSV converter 同樣可重跑。同 synthetic run_id 同內容略過且不新增歷史列，不同內容 exit 2。其他 history、單位、原文保留與 pending 規則不變。

無本工具 ownership／manifest 的 entry 不會被自動認領。應急工具或人工留下的半份資料，仍走[部分接手](#部分接手)。不要把手動備份命令當成工具 resume；兩者的 hash 清單格式不同。

## 名詞

- **來源**：本次的 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`，包括隱藏檔、空目錄與未知檔。
- **備份**：平行備份根裡與來源相同相對路徑的副本，hash 與來源逐檔相同。後續一律從備份讀，來源保持不動直到第 9 步。
- **報告**：`manual-report.json`，每個來源檔與每個未轉欄位一列，記去向與狀態。它是交接清單，不是工具的 resume manifest。
- **狀態**：`verified` 表示轉換目的已讀回核對；`preserved` 表示原樣保留且已核對，不宣稱已轉格式或理解欄位語意。`temporary-pending` 表示仍待補齊必要證據；`permanent-pending` 表示缺證據無法轉換，列出備份內的位置與缺的證據。
- **證據**：舊檔裡明寫的值。路徑日期、mtime、檔名、目前 setup 與目前裝置都不算證據。缺證據就列 pending，不推論、不補零。

## 1. 固定路徑與授權

停止會寫入這些目錄的 GUI、Notebook 與 agent。向使用者取得本次讀、寫與最後刪除的授權；沒有刪除授權時，做到第 8 步為止。
命令使用 Bash 與 GNU coreutils、findutils。範例值全部換成該機確認過的值。

```bash
set -euo pipefail
export REPO=/absolute/path/to/Qubit-measure-gui
export CHIP=TestChip QUB=Q0 NAME=migrated-Q0
export RESULTS="$REPO/result" DATABASE="$REPO/Database"
export OLD_R="$RESULTS/$CHIP/$QUB" OLD_D="$DATABASE/$CHIP/$QUB"
export NEW_R="$RESULTS/$NAME" NEW_D="$DATABASE/$NAME"
export BACKUP_ROOT="$(dirname "$REPO")/$(basename "$REPO")-backup"
export BR="$BACKUP_ROOT/result/$CHIP/$QUB" BD="$BACKUP_ROOT/Database/$CHIP/$QUB"
export REC="$BACKUP_ROOT/migration-records/$CHIP/$QUB"
realpath -m -- "$OLD_R" "$OLD_D" "$NEW_R" "$NEW_D" "$BR" "$BD" "$REC"
```

逐項確認，全部成立才往下：

- `CHIP`、`QUB`、`NAME` 各是單一路徑段，不含 `/`、`.`、`..`。
- `BACKUP_ROOT` 在 repo 外。以下手動全新備份要求 `BR`、`BD`、`REC` 不存在。本工具已擁有的備份直接走工具重跑；其他既有備份走[部分接手](#部分接手)。
- 手動全新遷移要求 `NEW_R`、`NEW_D` 不存在。這是遷移保護目的資料的條件，不是 `ResultEntry.create` 的規則；一般 create 只檢查 Result。本工具擁有的 entry 可直接重跑，其他既有 entry 走部分接手，不刪掉重來。
- 新 entry 不包住、也不透過 symlink 指進任一來源。
- `find "$OLD_R" "$OLD_D" -type l -o ! -type d ! -type f` 沒有輸出。有輸出就停下，向使用者確認實體來源。
- `OLD_D` 不存在時，記下並取得使用者確認；之後只備份 `OLD_R`，不建立空 Database 充數。

## 2. 備份到平行備份根並驗證

本節只建全新的手動備份，不能對本工具的既有備份紀錄執行。手動清單是 `sha256sum` 文字，工具的同名 `.sha256` 是含檔案與目錄的 typed JSON，兩者不互換。

```bash
hash_tree() {
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum)
}
test -d "$OLD_R" && test -d "$OLD_D"
test ! -e "$BR" && test ! -e "$BD" && test ! -e "$REC"
mkdir -p -- "$REC" "$(dirname "$BR")" "$(dirname "$BD")"
hash_tree "$OLD_R" > "$REC/result.sha256"
hash_tree "$OLD_D" > "$REC/Database.sha256"
cp -a -- "$OLD_R" "$BR"
cp -a -- "$OLD_D" "$BD"
(cd "$BR" && sha256sum --quiet -c "$REC/result.sha256")
(cd "$BD" && sha256sum --quiet -c "$REC/Database.sha256")
cmp "$REC/result.sha256" <(hash_tree "$BR")
cmp "$REC/Database.sha256" <(hash_tree "$BD")
diff -qr -- "$OLD_R" "$BR"
diff -qr -- "$OLD_D" "$BD"
cmp "$REC/result.sha256" <(hash_tree "$OLD_R")
cmp "$REC/Database.sha256" <(hash_tree "$OLD_D")
```

完成條件：每個命令 exit 0，`cmp`、`diff` 沒有輸出。最後兩行確認備份期間來源沒有變動。
`sha256sum -c` 抓不到多出的檔案，所以還要比完整清單。

接著在 `$REC/notes.md` 寫下路徑、操作者、授權範圍與目前 checkout 的 commit。之後的工作檔都放 `$REC`。

**中斷時：** 保留既有備份，用 `$REC` 的清單重驗。驗證失敗的半份備份不能當清理依據。

**來源已被應急工具搬走一部分時：** 現在複製的只是殘存部分，不算完整備份。先找回搬移前的完整副本，或用原始 hash 清單從新舊兩端逐檔補回備份內的原相對路徑，並記錄補回來源。補不齊就停在這裡，不進清理。

## 3. 盤點每個舊檔的去向

```bash
find "$BR" -type f -printf '%P\n' | LC_ALL=C sort > "$REC/result-files.txt"
find "$BD" -type f -printf '%P\n' | LC_ALL=C sort > "$REC/Database-files.txt"
find "$BR/exps" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort
```

最後一行列出的每個 `exps/<sub>` 就是一個新 point，`<sub>` 原樣當 point label，例如 `070716_0.500`，不從名稱解讀磁通或時間。
label 不合法時，由使用者決定新名，報告記舊名、新名與理由。沒有 `exps/` 時，改列實際存在的舊直屬 point 目錄。

每個舊檔在報告中都要有一列。下表是去向：

| 舊檔 | 新位置 | 做法 |
| --- | --- | --- |
| 根層 `meta_info.json` | `setup.yaml` | 第 4 步 |
| 根層 `params.json` 的 `fluxdep_fit`，或更早的 `params.EJ/EC/EL`、`half flux`、`period` | `predictors/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的 `dispersive`，以及 `fluxdep_fit.plot_transitions` 或舊 `allows` | `dispersive/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的 `t1_curve_fit` | `t1_curve/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的未轉原文 | `components.<Q>.ext.legacy_params.<原key>` | 第 4 步。只留未轉 leaf；`params.json` 本身不複製進新 entry。 |
| 根層 `module_cfg.yaml` | entry 的 setup library | 第 6 步 |
| `exps/<sub>/meta_info.json` | `points/<sub>/point.yaml` | 第 6 步 |
| `exps/<sub>/module_cfg.yaml` | `points/<sub>/module_cfg.yaml` | 第 6 步 |
| `exps/<sub>/image/*.png` | `points/<sub>/figures/<原檔名>` | 第 7 步 |
| 舊直屬 `<sub>/{meta_info.json,module_cfg.yaml}` | 同名 point | 與 `exps/<sub>` 同名時停下，列出兩個來源，不合併。 |
| exps 以外的圖 | `result/<name>/figures/<原相對路徑>` | 第 7 步 |
| `samples.csv` | `samples/<Q>.jsonl` | 第 7 步。有列轉不了時，原檔另外原樣放 `result/<name>/samples.csv`。 |
| `arb_waveforms/**`（result 或 Database） | `Database/<name>/arb_waveforms/**` | 第 7 步 |
| `data/fluxdep/spectrums.hdf5` | 新 entry 同相對位置；未轉原文另存 `data/fluxdep/spectrums.legacy.hdf5` | 第 7 步的 processed spectrum 規則。不能整檔當 generic raw copy。 |
| `autofluxdep_runs/**`、其他不支援的 result 檔 | `result/<name>/<原相對路徑>` | 第 7 步，原樣複製。 |
| Database 的 Labber 檔 | 原檔一律放 `Database/<name>/Labber/<原相對路徑>`；能轉的另產 `Database/<name>/runs/<experiment>_<run_id>/data.h5` | 第 8 步 |
| 其他 Database 檔 | `Database/<name>/<原相對路徑>` | 第 7 步，原樣複製。 |

`<Q>` 是舊 qubit 對應的新 component 名。找不到唯一對應時，所有寫到 `<Q>` 的項目整批列 permanent-pending，不任選一個。

報告放 `$REC/manual-report.json`，最小格式：

```json
{
  "mode": "manual-handoff",
  "entry": "migrated-Q0",
  "backup_root": "/abs/Qubit-measure-gui-backup",
  "items": [
    {
      "source": "result/TestChip/Q0/params.json#fluxdep_fit.timestamp",
      "source_sha256": "<actual SHA256 of params.json>",
      "destination": "result/migrated-Q0/predictors/Q0.json#source.created_at",
      "status": "permanent-pending",
      "reason": "timestamp has no time zone"
    }
  ],
  "unfinished_publications": [],
  "temporary_pending": []
}
```

`source` 是 repo 內的相對路徑，也就是備份根內的相對路徑。衝突列兩個來源、兩個值、單位與採用理由。
完成條件：每個來源檔、每個未轉欄位都有一列，不能只寫總數。

## 4. 建 entry 並寫 setup

用公開 API 建骨架，不手填 UUID。這一步只寫檔，不接硬體。

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

印出的 UUID 與兩個路徑記入 `$REC/notes.md`。此時只建立 `$NEW_R`，`$NEW_D` 只是後續輸出位置，create／open 都不檢查或建立它。第 7 步的 Database 檔案複製或第 8 步的 native publication 才建立所需目錄。部分接手時跳過這段，沿用既有 UUID。

接著編輯 `$NEW_R/setup.yaml`，欄位依[正式 component 定義](../zcu_lab/components/)：

1. **meta_info → 正式欄位。** 依下表寫入。meta 與 params 衝突時採 meta，報告列兩來源。

   | 舊 key | 新 logical path |
   | --- | --- |
   | `q_f`、`qf_w`、`t1`、`t2r`、`t2e` | `Q.freq`、`Q.kappa`、`Q.t1`、`Q.t2r`、`Q.t2e` |
   | `r_f`、`rf_w` | `R.freq`、`R.kappa` |
   | `qub_0_1_ch`；沒有它時用 `qub_ch` | `Q.wiring.ch` |
   | `lo_flux_ch`、`res_ch`、`ro_ch` | `Q.wiring.lo_flux_ch`、`R.wiring.ch`、`R.wiring.ro_ch` |
   | `flux_unit`、`cur_A` | `general.ext.flux_unit`、`general.ext.flux_value` |
   | `flx_half`、`flx_int`、`flx_period` | `Q.flux_half`、`Q.flux_int`、`Q.flux_period` |
   | `local_flux_unit`、`jpa_flux_unit`、`best_jpa_flux` | `Q.flux_unit`、`J.flux_unit`、`J.flux` |
   | `t1err`、`t2r_err`、`t2e_err` | `metadata[Q.t1 / Q.t2r / Q.t2e].stderr` |
   | `pi_gain`、`pi_len`、`pi2_gain`、`pi2_len` | 有對應的轉換後 module 時寫進該 module；沒有時存 `Q.ext.<舊 key>` |

   Q、R、J 是已確認的實際 component 名。缺 R 或 J 的實體對應時，相關欄位列 pending，不新增虛構元件。
   未映射的 meta key 以原名原值放在該 setup 或 point 的 `general.ext` 根層，包含 scalar、nested object、list 與 null。來源 key 若是 `legacy_meta_info` 或 `legacy_module_cfg`，放在 `general.ext.legacy_meta_info` 的同名 key。例如 `legacy_module_cfg: [old]` 寫成 `general.ext.legacy_meta_info.legacy_module_cfg: [old]`，不覆寫容器。報告逐項標 `preserved`，不因未知欄位列 pending。`source_keys` 與 `destination_keys` 是 literal key 序列，含點的原 key 仍是一個 segment，保留 row 的 `new_path` 是 null。

   如果未知原 key 撞本 context 正式投影的實際 ext 目的，先保留正式值。例如 `cur_A: 0.4` 已寫入 `general.ext.flux_value`，未知的 `flux_value: 0.9` 就存到 `general.ext.legacy_meta_info.flux_value`。若該未知原值也是 0.4，按這個正式目的去重，報告用 `remove` 列出 `general.ext.flux_value`。其他位置碰巧出現相同數字，不構成去重理由。Point 繼承 setup 已成功正式投影的 ext 目的後，也按此規則保護該目的。Point 的撞名原文不同值就存入該 point 的 `legacy_meta_info[原key]`，同值按對應繼承目的去重。必須能追到 setup 的成功正式 projection，不能只因 ext 目前有同名值就當成正式欄位。Point 若明示正式來源，仍依同檔優先與原子分組清除規則處理。

   `qub_0_1_ch` 與 `qub_ch` 都存在時，正式 wiring 採前者。兩值不同就把落選的 `qub_ch` 原值放入 `general.ext.legacy_meta_info`；同值才去重。舊 remove 表中的 `readout_f`、`flx_bias`、`mA_c`、`mA_e`、`resetf1_w`、`cur_jpa_A`、`fake_peak`、`log_scale` 也不能直接丟棄。`readout_f` 的正式對應是 resonator 的 `freq`，`flx_bias` 的正式對應是 `flux_bias` 寫入的 qubit `ext.flux_bias`。只有同一份 meta 的對應正式來源成功寫入且同值時，才不另存原文，報告用 `remove` 指出正式檔案與欄位。不同值、未成功轉換或沒有可靠對應時，原名原值放入 `general.ext.legacy_meta_info`。不要拿繼承值或其他欄位碰巧相同的數字當去重依據。歷史 key 全表在 [storage_mapping.py](../migrations/storage_mapping.py)。

2. **單位照抄，不換算。** 頻率 MHz、時間 us、能量 GHz。磁通數字分兩組：`flux_unit` 管 global 組（`cur_A`、`flx_half`、`flx_period`、`flx_int`、`flux_bias`），單位寫到 `general.ext.flux_unit`；`jpa_flux_unit` 管 `best_jpa_flux`，單位寫到 `J.flux_unit`。`local_flux_unit` 只照抄成 `Q.flux_unit`，不當任何數字的單位。每組依序取單位：
   1. 同一份 meta 明寫的單位。
   2. entry 已知的明寫單位：根層 meta_info 的同一個 key；沒有時用 `params.json` 根層的 `flux_unit`。JPA 只認 `jpa_flux_unit`，不借 qubit 的單位，所以跳過 `params.json` 這一項。
   3. 使用者宣告的假設單位，報告標「假設單位」。本專案的資料：global 組用 `mA`，JPA 組用 `V`。
   4. 以上都沒有：數字不寫入，列 pending。

   同一組、同一層的兩個明寫單位互相衝突時（例如根層 meta 與 `params.json` 不同），列 pending，不擇一。單位不符合欄位型別時（`J.flux_unit`、`Q.flux_unit` 只接受 A 或 V），不換算，整組列 pending。數字與單位一起寫入或一起 pending。被阻擋的原 source key/value 存入該 context 的 `general.ext.legacy_meta_info`，報告的 pending 指向此位置，保留 row 仍標 `preserved`。來源的未知單位或衝突單位保留原文，不寫成生效單位，也不改成解析後的 null。只保留來源實際存在的 key，不另存已轉換的值或工具補出的 evidence key。`local_flux_unit` 不被其他組借用，若 owner 不接受它，單獨 pending 並保留原文。報告記下實際採用的是哪份檔的單位。

3. **params 原文 → `legacy_params`。** `params.json` 的每個區塊連同內含 metadata 原文寫進 `components.<Q>.ext.legacy_params.<區塊名>`。
   第 5 步轉成格式檔的區塊也照樣保留原文。`project`、`name`、`schema_version` 等識別資料也原文保留，不冒充 entry UUID。

範例（合成值）：

```yaml
format: zcu.parameter-container
format_version: '2.0'
general:
  entry_id: '54cd2da2-e8cd-49d5-a414-4ec40eb7eb14'
  created_at: '2026-10-20T08:00:00Z'
  description: 'Manual migration of TestChip/Q0'
  ext: {flux_unit: mA, flux_value: 0.5}
components:
  Q0:
    kind: qubit/fluxonium
    freq: 5000.0
    ext:
      legacy_params:
        dispersive:
          params: {g: 0.12, bare_rf: 7.1}
          timestamp: '2024-05-01 10:00'
metadata:
  Q0.freq: {source: 'meta_info.json#q_f'}
```

完成條件：setup 的每個值都能指回一個舊 key，pending 的欄位不出現在 setup 的可用數值裡。

## 5. 寫 entry 層格式檔

`fluxdep_fit`、`dispersive`、`t1_curve_fit` 各寫成 entry 層的一個 JSON 檔，檔名是 `<Q>.json`。三者共用同一個外框：

```text
{"format": "...", "version": 1, "params": {...}, "source": {"run_id": null, "created_at": null}}
```

- 頂層只有表列的 key（predictor 另加 `model`、`flux_unit`）。`version` 是嚴格整數 `1`。
- `source.run_id`：遷移一律 `null`。
- `source.created_at`：只有原區塊 `timestamp` 是 `YYYY-MM-DDTHH:MM[:SS[.ffffff]]` 加 `Z` 或 `±HH:MM` 的形式（例：`2026-10-20T08:00:00Z`）時才照抄原字串；否則 `null`，報告列 pending，附備份內原文位置。entry UUID、mtime 與遷移時間都不能代填。
- 未知數值填 `null` 或省略，不補 0。已知數值必須是有限數字，不接受 bool 或數字字串。
- 格式檔只放在 entry 層，不複製到 points。`legacy_params` 只留未轉原文，不再存已轉 leaf；完整來源仍在備份裡。

寫完每個檔都要用 owner 讀回一次：predictor 用 `zcu_tools.resources.predictor_file.load_predictor_file(path)`，dispersive 用 `zcu_tools.resources.dispersive_file.load_dispersive_file(path)`，T1 用 `zcu_tools.resources.t1_curve_file.load_t1_curve_file(path)`。它們會驗證欄名、數值與 source。見 [resources README](../lib/zcu_tools/resources/README.md)。

### predictor：`predictors/<Q>.json`

舊 Fluxonium fit 的 `fluxdep_fit.params` 有 `EJ` 或 `EL` 時，支援欄照轉，extra key 原名原值留在 `component.ext.legacy_params`。只有 `EC` 加陌生 key 或只有陌生 key 時，不建 predictor，原樣保留並報不支援模型，不列 pending。已帶 `model` 的 fit 仍列 pending，不推論模型。原本支援的乾淨部分參數或空參數容器仍可轉換。非法已知值不寫成成功數值，原文保留並列 pending。

| 舊值 | predictor 欄位 |
| --- | --- |
| `fluxdep_fit.params.EJ`、`EC`、`EL` | `model: "fluxonium"`，`params.EJ`、`params.EC`、`params.EL`，GHz |
| `fluxdep_fit.flux_half`、`flux_period` | `params.flux_half`、`params.flux_period`，不換算 |
| （舊 predictor 的預設） | `params.flux_bias: 0.0` |
| `fluxdep_fit.flux_int` | 不寫入；它只用於舊 alignment，原文留在 `legacy_params` |
| params 根層 `flux_unit` 或 `fluxdep_fit.flux_unit` | `flux_unit`。只有一處時用該值；兩處相同時用該值；兩處衝突時寫 `null` 並列 pending；兩處都沒有時用使用者宣告的假設單位並標「假設單位」，沒有宣告就寫 `null` 並列 pending |
| `fluxdep_fit.timestamp` | `source.created_at`，規則見上 |
| 更早的根層 `params.EJ`、`EC`、`EL` | `params.EJ`、`params.EC`、`params.EL`，不改數字 |
| 更早的根層 `half flux`、`period` | `params.flux_half`、`params.flux_period`，不換算 |
| 更早形狀的根層 `timestamp` | `source.created_at`，仍須符合上列含時區形式 |

更早的完整形狀如下。這裡的 `params` 是舊檔根層的 key，不是 `fluxdep_fit.params`：

```json
{"name": "old author", "params": {"EJ": 6.27, "EC": 1.41, "EL": 0.8}, "half flux": 0.25, "period": 1.5, "allows": {"sample_f": 9.7, "transitions": [[0, 1]], "mirror": [[1, 2]]}}
```

手動轉換時先看有沒有 `fluxdep_fit`。兩種形狀同時存在，就用較新的 `fluxdep_fit` 寫 predictor。逐項比較根層舊值與它對應的新來源；同值不再抄進 ext，不同值以原名留在 `legacy_params`。例如舊 `half flux=0.25`、新 `fluxdep_fit.flux_half=0.3`，predictor 寫 `0.3`，ext 留 `half flux: 0.25`。`name`、`project`、`schema_version` 等 metadata 不改名，也不冒充 predictor 欄位。

腳本報告使用 `source_keys` 定位根層來源，`compared_source_keys` 定位同檔較新來源。同值列 `action="remove"`；不同值列 `action="preserved"`，`destination_keys` 指向 ext，reason 列出雙值。這兩種狀況不因來源不同就列 pending。

`cur_A` 這類欄名、meta_info 的單位、目前裝置的單位都不能當 predictor 的單位證據。

```json
{
  "format": "zcu.predictor",
  "version": 1,
  "model": "fluxonium",
  "params": {"EJ": 6.27, "EC": 1.41, "EL": 0.80, "flux_half": 0.001, "flux_period": 0.002, "flux_bias": 0.0},
  "flux_unit": "A",
  "source": {"run_id": null, "created_at": null}
}
```

### dispersive：`dispersive/<Q>.json`

`dispersive` 的 `g`、`bare_rf` 寫進 `params`；`fluxdep_fit.plot_transitions` 寫進 `params.plot_transitions`。
更早的根層 `allows` 也寫到 `params.plot_transitions`，原 key 與內容不變。`sample_f` 不改成 `r_f`，`transitions`／`mirror` 的內容不改。Owner 把這些內容當 annotations 保存，不代表新舊分析公式等價。
只有 transition 內容、沒有 `dispersive` 區塊時也要產檔，`g`、`bare_rf` 可以省略或寫 `null`。
只有 `allows` 來源時，不在 `legacy_params` 再存一份。若同時有 `fluxdep_fit.plot_transitions`，後者優先寫正式檔。按對應 key 比較 `allows`，同值 leaf 去重，不同或無對應 leaf 原名原值留在 `legacy_params.allows`，清空的父分支移除。腳本沿用上述 typed 比較報告。
`g`、`bare_rf`、`plot_transitions.r_f` 的單位是 GHz；`plot_transitions` 其他欄位原文照抄。
`source.created_at` 取 `dispersive.timestamp`；只有 `fluxdep_fit.plot_transitions` 時取 `fluxdep_fit.timestamp`；只有根層 `allows` 時取根層 `timestamp`。都必須符合含時區形式，否則寫 null 並列出備份定位，不用檔案時間代填。

```json
{
  "format": "zcu.dispersive",
  "version": 1,
  "params": {"g": 0.12, "bare_rf": 7.1, "plot_transitions": {"r_f": 7.1}},
  "source": {"run_id": null, "created_at": null}
}
```

### T1 曲線：`t1_curve/<Q>.json`

`t1_curve_fit` 的 `params`、`stderr`、`bounds`、`fixed`、`free`、`cost`、`init`、`reduced_chi2`、`success`、`message`、`residual_mode`、`loss`、`max_nfev` 原樣寫進 `params`。
`Temp` 的單位是 K；`Q_cap`、`Q_ind`、`x_qp` 沒有單位。`params.params` 裡出現的 key 是 active 參數，值為 `null` 表示 active 但未知。
`fixed`、`free`、`bounds`、`stderr`、`init` 只能引用 active 參數。

```json
{
  "format": "zcu.t1-curve",
  "version": 1,
  "params": {
    "params": {"Q_cap": null, "Temp": null},
    "stderr": {"Q_cap": null},
    "fixed": ["Temp"],
    "free": ["Q_cap"],
    "bounds": {"Q_cap": [null, null]},
    "cost": null
  },
  "source": {"run_id": null, "created_at": null}
}
```

完成條件：每個有內容的舊區塊都有對應的格式檔，或在報告中列 pending；每個 `null` 都有對應的報告列。

## 6. 建 points 與 module library

### point：先整份複製 setup，再套 point meta

setup 完成後才建 point。每個 point 都從完整的 `setup.yaml` 複製，所以 `general.ext.flux_unit` 這類 entry 層資訊會一起帶到 point。

```bash
export LABEL=070716_0.500
export POINT="$NEW_R/points/$LABEL"
test ! -e "$POINT"
mkdir -- "$POINT"
cp -- "$NEW_R/setup.yaml" "$POINT/point.yaml"
printf "format: zcu.module-library\nformat_version: '2.0'\n" > "$POINT/module_cfg.yaml"
date -u +%Y-%m-%dT%H:%M:%SZ
```

接著編輯 `point.yaml`：

1. `general.entry_id` 保留不動。它是複製來源 setup 的 UUID，不是 point 的身分。
2. `general.created_at` 改成剛印出的時間，也就是新 point 的建立時間。
3. 套用 `exps/<sub>/meta_info.json`：用第 4 步的對照表，point meta 有的值覆蓋 setup 繼承的值。
4. 單位依第 4 步的順序決定。point meta 沒寫單位時，沿用從 setup 繼承的單位，報告記「unit from root meta_info.json」。
   point meta 寫了與 setup 不同的單位時，移除同一組從 setup 繼承、但 point 沒覆寫的數字及其 metadata，只留 point meta 自己的數字，不讓 setup 的數字配上 point 的單位。

完成條件：每個 `exps/<sub>` 對應一個 point，各自有完整 components、metadata 與正確的 flux 單位。
`entry.list_points()` 只列出 `point.yaml`、`module_cfg.yaml` 兩檔都在的目錄，所以另外用 `find "$NEW_R/points" -mindepth 1 -maxdepth 2` 檢查未完成的骨架。

### module_cfg：轉成正式 Spec

舊 `module_cfg.yaml` 的 `modules`、`waveforms` mapping 不是新 library，只換 header 不算轉換。逐筆建立「舊 family、舊 key、新 canonical key、引用位置」的對照，轉成 `components.<Q>.{pulses,readout,reset,waveforms}.<name>`，再把 point 的 module slot 指到新 key，例如 `components.Q0.module.x180: Q0.pulses.pi_amp`。

1. 先重建已證實的歷史 shape。完整的舊 Pulse 至少有 ch、nqz、freq、gain、waveform；沒有其他 type 證據且缺 type 時才補 `type: pulse`，不按名稱猜。歷史 `Pulse.auto_fill` 的 phase、pre_delay、post_delay 預設都是 0.0，已有值保持不動。desc 與 mixer_freq 是正式欄位，不能丟掉。
2. 舊 `type: base` 轉成平鋪的 `readout/pulse`。pulse_cfg.ch 改名 gen_ch，pulse_cfg 的頻率、gain、phase、延遲、mixer_freq、waveform 保持原值；ro_cfg 的 ro_ch、ro_length、trig_offset 移到外層。外層 desc 是正式描述，內層 pulse_cfg.desc 沒有獨立對應，要留原文。沒有 ro_freq 時，ADC 使用 pulse_cfg.freq。明示 ro_freq 等於 pulse_cfg.freq 時共用正式 freq，不另存同值；不同時現行單一 freq 無法同時表達，整條原文保全，不建該條 Spec，pending 記兩個頻率。
3. 舊 `type: bath` 改為 `reset/bath`。cavity_tone_cfg、qubit_tone_cfg、pi2_cfg 是三個已知 Pulse 位置，分別補 type、套用上述 defaults，保留其 desc、mixer_freq、phase 等正式值。name、block_mode 等未表達欄位留原位置，不把整條成功 Spec 再存一份。
4. 用 [cfg_model](../lib/zcu_tools/cfg_model/README.md) 的 `partition_module_spec(raw)`、`partition_waveform_spec(raw)` 分離未知欄位，取得 typed spec 與 unrecognized 樹。輸入此時必須已是現行欄名與 discriminator。輸出仍經 `module_spec_from_raw`／`waveform_spec_from_raw` strict 驗證。Child key 用 `remap_module_references`、`remap_waveform_references` 改；expression 不求值，缺的必要欄位不猜。只在未知欄位中的 expression 原樣保全，不阻斷其餘正式輸出。
5. 把未表達欄位樹寫進該 context 的 `general.ext.legacy_module_cfg[原section][原name]`。未知 section 保持原 section 名。完整不支援或沒有 mapping 的條目才存整條原文，不列 pending。支援項缺欄位或 child mapping 仍 pending，原文也要在此可讀。只留未轉 leaf，移除空分支，不把成功值存兩份。透過 view.edit() 設定整個 general.ext mapping，可保留含點號的 literal key。例如未知 `modules.drive.old.key` 若原本 old.key 是單一 key，就寫 `legacy_module_cfg: {modules: {drive: {old.key: 原值}}}`，不要拆開它。報告的 preserved row 用 literal source_keys／destination_keys、new_file 與 new_path=None 定位。
6. 用 owner 提交並讀回：

```python
from zcu_tools.cfg_model import waveform_spec_from_raw
spec = waveform_spec_from_raw(raw_waveform)
with entry.module_library(label).edit() as draft:
    draft.set_waveform("Q0.waveforms.drive", spec)
print(entry.module_library(label).snapshot().waveform_spec("Q0.waveforms.drive"))
```

`entry` 依第 9 步的方式註冊 component models 後開啟。provenance 由 owner 產生，不手寫事件或 run_id。
一般讀取 library 時會自動備份並移除壞掉的條目，所以先在工作副本試轉候選，不拿真檔試錯。

**根層 `module_cfg.yaml`** 用同樣規則經 `entry.module_library()` 寫入 setup library。Point 自己的庫用 `entry.module_library(label)`，新 point 不自動複製 setup library。Slots 只指向成功發布的 Spec，按 mapping 次序選第一個成功候選；不能用 pending 或只保全原文的名稱建立引用。

完成條件：每筆舊 module 和 waveform 都已經 owner 提交並讀回，或在新 context 的 legacy_module_cfg 可讀且報告標 preserved。真正缺轉換證據的支援項另列 pending。空 library 本身合法，但不能說成「舊 library 已全部轉換」。

## 7. 複製圖、樣本與保全檔

所有複製都用這個 helper：目的不存在才複製；已存在且 bytes 相同算完成；不同就停下並記衝突。

```bash
copy_checked() {
  local src=$1 dst=$2
  test -f "$src" && test ! -L "$src" && test ! -L "$dst"
  mkdir -p -- "$(dirname "$dst")"
  if test -e "$dst"; then cmp -- "$src" "$dst"; else cp --no-clobber -- "$src" "$dst" && cmp -- "$src" "$dst"; fi
  sha256sum -- "$src" "$dst"
}
shopt -s nullglob
for point_src in "$BR"/exps/*/; do
  label=$(basename "$point_src")
  test -f "$NEW_R/points/$label/point.yaml"
  for png in "$point_src"image/*.png; do
    copy_checked "$png" "$NEW_R/points/$label/figures/$(basename "$png")"
  done
done
```

- **exps 以外的圖**：`copy_checked "$BR/<相對路徑>" "$NEW_R/figures/<相對路徑>"`，保留原檔名與目錄。
- **`arb_waveforms/**`**：`$BR/arb_waveforms/` 與 `$BD/arb_waveforms/` 的每個檔複製到 `$NEW_D/arb_waveforms/`，保留相對路徑。兩個來源落到同一目的而 bytes 不同時，停下並列出兩邊的 hash。
- **不支援的檔**：沒有轉換規則的舊檔原樣複製到新 entry 的對應位置。result 那邊的 `$BR/<相對路徑>` 複製到 `$NEW_R/<相對路徑>`；`exps/<sub>/` 裡的檔複製到 `$NEW_R/points/<sub>/<sub 內相對路徑>`；Database 那邊的 `$BD/<相對路徑>` 複製到 `$NEW_D/<相對路徑>`。報告註明它不是新格式資料。對應位置已被新格式的檔或目錄佔用時（例如 `setup.yaml`、`points/`、`records/`、`runs/`），停下列衝突，不覆寫。
- **`samples.csv`**：由使用者指定值欄與單位，例如值欄 `calibrated mA`、單位 `mA`，再用 sample owner 轉換：

  `entry` 依第 9 步的方式開啟。

```python
import os
from migrations.sample_csv import convert_sample_csv
report = convert_sample_csv(
    [os.environ["BR"] + "/samples.csv"], entry=entry, component="Q0",
    dev_value_column="calibrated mA", source_unit="mA",
)
print(report)
```

  已知量測欄按公開 converter 改名，座標與明示單位沿 sample owner 規則轉換。未知欄位原名原字串留在 JSONL 根層，包含空字串及前後空白，不猜型別。例如 `comment=Manual Added` 不改成 `Tcomment`，也不讓整列 pending。

  原欄名碰到 `run_id`、`point`、`source`、`at` 或 canonical 量測名稱時，原值留在 `legacy_fields` mapping 的同名 key。來源本身的 `legacy_fields` 欄也存在 mapping 內。例如 `T1 (us)` 轉成數字 `t1_us`，原欄 `t1_us` 存到 `legacy_fields["t1_us"]`。

  將回傳的 `preserved_fields` 逐筆記入手動報告，標為 `preserved`。每筆有來源檔、資料列號、原欄名與原字串、synthetic run_id 及 `destination_keys`。列號從 1 起算且不含 header，目的 keys 是 literal JSON key 序列，不用點號拆原欄名。直接 convert 的 `output_path` 是 JSONL 位置；storage migration 報告的 `preserved_sample_fields` 為每筆欄位另附 `output_path`，並存入 manifest，resume 不丟棄。

  回傳的 pending 列也逐列記入報告。Legacy CSV 缺使用者宣告的值欄或單位時，相關列仍 pending，完整 CSV 另保留在新 entry 的 `samples.csv`；v2 CSV 自帶 `dev_value/dev_unit`，不需要 legacy 宣告。非法資料須列出錯誤與定位，不能記成已轉換。用法細節見 [migrations README](../migrations/README.md#sample-csv--jsonl)。

完成條件：每對 hash 相同；每張 `exps` 圖都掛在正確的 point。helper 中斷可能留下半份目的檔，記入 `unfinished_publications`，從備份核對後再補。

### Processed spectrum

舊 `data/fluxdep/spectrums.hdf5` 不能只複製到新 entry。從 `$BR` 的備份讀取，正式目的為 `$NEW_R/data/fluxdep/spectrums.hdf5`，剩餘原文目的為同目錄 `spectrums.legacy.hdf5`。現有 processed publication 只重用 matching manifest 的 prepared/published ownership 且 hash 一致的目的，不認領任意同名檔。遇到衝突不覆寫。手動接手時先保存已有目的及 hash，依原文逐項比對，不按同名檔猜測已完成。

以每個原 group 名建立一個 `SpectrumResult`。欄位依下表轉換，陣列不改值或 dtype，不按數字量級換算單位。

| 舊位置 | 正式位置／公式 |
| --- | --- |
| `mA_c` | `flux_half` |
| `period` | `flux_period` |
| `mA_c`、`period` | `flux_int = mA_c + 0.5 * period` |
| `spectrum/mAs`、`points/mAs` | 同 section 的 `dev_values` |
| `spectrum/fpts`、`points/fpts` | 同 section 的 `freqs`，不再做 Hz→GHz 換算 |
| 各 section 的 `mAs`、group 的 `mA_c/period` | `fluxs = (mAs - mA_c) / period + 0.5` |
| `spectrum/data` | `spectrum/signals`，按下述軸向規則處理 |
| 字串 group attr `type` | 正式 group attr `type`，原字串不改 |

`mAs/fpts/data` 格式曾使用兩種軸向。若兩條軸長度不同，shape 為 `[len(mAs), len(fpts)]` 就直接使用，反過來就轉置。現行 `signals` 的第一維是 device，第二維是 frequency。平方矩陣缺歷史版本證據時不能猜軸向，整個 group 先留 companion，報 permanent-pending 並指出缺少版本證據。其他 group 照常轉換。中繼格式的 `flx_half/flx_int/flx_period` 改名為 `flux_half/flux_int/flux_period`，已有的 `dev_values/fluxs/freqs/signals` 直接沿用。非字串 `type` 留 companion，不捏造 OneTone／TwoTone。

正式檔交給現有 owner，不手寫其 HDF5 layout：

```python
from zcu_tools.analysis.fluxdep.io import dump_spectrums, load_spectrums

# converted maps each original group name to the SpectrumResult built above.
dump_spectrums(str(new_r / "data/fluxdep/spectrums.hdf5"), converted)
restored = load_spectrums(str(new_r / "data/fluxdep/spectrums.hdf5"))
```

逐一核對 restored 的 scalar、兩條軸、picked points、signals 數值及 dtype。另用 h5py 的 `File(..., "x")` 建 companion，只寫尚未轉換的 datasets、groups 與 attrs。每項都沿原 hierarchy，dataset 可用 h5py `Group.copy` 保留 dtype 與 attrs。已轉 dataset 若仍有 attrs，改建同路徑 group 承載 attrs，不再存 data。Attr 用原名、原值與原 dtype 建立。已正式轉換的 `type` 不重存。沒有剩餘原文就不建 companion。

Companion 是 migration 保留的原文，不是 SpectrumResult 格式，正常 runtime 不讀它。用 h5py 讀回逐項核對原值、shape、dtype 與 attrs，不能只核對整個來源備份存在。報告逐項分成正式轉換、`preserved` 及缺證據 pending。工具的 `key_mappings` 保留 literal source/destination keys，attr 定位為 object path 加原 attr 名，display key 以 `attribute:` 開頭。保留 row 的 `new_path` 為 null，`new_file` 指 companion。手動報告也記下同樣位置及正式／companion 各自的 SHA256。兩檔都核對後才算這項完成。

## 8. Labber 檔轉 native

Labber 與 native 都是 HDF5，但格式不同，把 `.hdf5` 改名成 `data.h5` 不會得到 native。

**必要證據**齊全才轉：experiment tag 與 cfg_type、完整歷史 cfg、variable、axes 與 signal 的名稱、單位、dtype 和 shape。缺任何一項就整檔列 permanent-pending，原檔仍用 `copy_checked` 複製到 `$NEW_D/Labber/<原相對路徑>`，和已轉換的 Labber 放在一起。

開始時間與 run_id 缺了也照樣轉：

- 沒有可證實的開始時間：`started_at=None`、`completion="unknown"`、`finished_at=None`。
- 沒有可信的 run_id：用 `resources.entry.new_run_id()` 分配，報告標「遷移賦值」。ID 裡的時戳是分配當下的時間，不能拿來填 `started_at`。
- 有可證實的時間或可信 ID 就保留原值；完成狀態只用證據支持的 complete、partial 或 stopped。

| 證據 | 寫到 |
| --- | --- |
| 來源相對路徑與 SHA256 | 報告；必須與備份清單一致 |
| tag、cfg_type、schema_version | 對應明確的 cfg model 與 disk schema |
| 完整歷史 cfg | `CfgSnapshot(values, cfg_type, schema_version)`，不用目前的預設值補 |
| variable、axes、signal | `VariableSchema`；axes 由內到外排列，array shape 是 axes 長度的反序 |
| 時間與完成狀態 | `RunMetadata` |
| entry、point、description、roles | `RunSnapshot`；沒有證據時 point 寫 None |
| 軟體、主機、SoC | `SoftwareProvenance`；未知寫 None 或空 mapping |
| 原 Labber 檔的新位置 | `RunMetadata.labber_path`，寫 `Labber/<原相對路徑>` |
| Labber `Instrument config` | `device_snapshot`。轉換對照由遷移工具提供，land 前手動寫 None，報告列 temporary-pending |

型別見 [datafile owner](../lib/zcu_tools/datafile/README.md) 與 [native models](../lib/zcu_tools/datafile/native_models.py)。
分配 run_id 前，先把「來源、hash → run_id」對照寫進 `$REC`。中斷後沿用同一份對照，不重新分配。

Native 路徑由 `SaveLayout` 依 evidence 的 experiment 與完整 run_id 計算，experiment 中的 `/` 轉 `_`，dry-run 與 resume 使用同一規則。原 Labber 檔仍保留原名與相對路徑，不改成新 Save 的檔名。

以下是合成範例。`synthetic_demo`、`DemoCfg` 不是 repo 已註冊的實驗；真檔要換成確認過的 `(tag, cfg_type, schema_version)`、完整歷史 cfg 與正式 schema。

```python
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID
import numpy as np
from migrations.legacy_labber import load_legacy_labber_payload
from zcu_tools.datafile import (
    AxisSchema, CfgSnapshot, DataVariable, RunMetadata, RunSnapshot,
    SoftwareProvenance, VariableSchema, load_run_data, save_run_data,
    validate_experiment_payload,
)
from zcu_tools.resources.entry import ArtifactKey, SaveLayout, new_run_id

bd = Path(os.environ["BD"])
source = bd / "2024/05/01/synthetic.hdf5"
relative = source.relative_to(bd).as_posix()
digest = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
identity_path = Path(os.environ["REC"]) / f"run-id-{digest[:12]}.json"
if identity_path.exists():
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    assert identity["source"] == relative and identity["source_sha256"] == digest
else:
    identity = {"source": relative, "source_sha256": digest,
                "run_id": new_run_id(), "run_id_origin": "遷移賦值"}
    with identity_path.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, ensure_ascii=False, indent=2)

schema = (VariableSchema(
    variable=DataVariable("response"),
    axes=(AxisSchema(name="frequency", unit="Hz", dtype=np.dtype("float64")),),
    signal_name="signal", signal_unit="V", signal_dtype=np.dtype("complex128"),
),)
metadata = RunMetadata(
    run_id=identity["run_id"], experiment="synthetic_demo",
    started_at=None, finished_at=None, completion="unknown",
    labber_path=f"Labber/{relative}",
    snapshot=RunSnapshot(
        entry_id=UUID("54cd2da2-e8cd-49d5-a414-4ec40eb7eb14"),
        entry_name="migrated-Q0", point="070716_0.500",
        description="Synthetic example only", roles={"qubit": "Q0"},
    ),
    provenance=SoftwareProvenance(
        software_versions={}, git_commit=None, git_dirty=None, qick_version=None,
        soc_fingerprint=None, hostname=None, simulation_id=None,
    ),
    device_snapshot=None,
)
layout = SaveLayout(
    result_path=Path(os.environ["NEW_R"]), database_path=Path(os.environ["NEW_D"]),
    experiment=metadata.experiment, run_id=metadata.run_id,
    point=metadata.snapshot.point, saved_at=datetime.now(timezone.utc),
)
outputs = layout.outputs(ArtifactKey(section="run", name="data", member="data"))
destination = next(output.path for output in outputs if output.format == "data_h5")
payload = load_legacy_labber_payload(source, schema=schema)
validate_experiment_payload(payload, schema=schema)
destination.parent.mkdir(parents=True, exist_ok=True)
save_run_data(destination, payload, metadata,
              cfg=CfgSnapshot(values={"reps": 10}, cfg_type="DemoCfg", schema_version="1.0"))
validate_experiment_payload(load_run_data(destination).payload, schema=schema)
```

寫入後：

1. 用該實驗的 `load_run` 讀回，逐欄比對 cfg、tag、時間、snapshot、axes、signal 與原 payload。typed metadata 用 `experiment.native_persistence.load_run_with_metadata(path, spec=...)` 核對。只接受 complete 的分析入口會拒絕 unknown，這不代表 native 無效。
2. 讀回通過後，`copy_checked "$BD/<相對路徑>" "$NEW_D/Labber/<相對路徑>"`，保留原檔名與 bytes。
3. 報告記錄 run_id 與它的來源（「遷移賦值」或證據）、native 路徑、Labber 路徑、原檔 hash，以及哪些時間未知。

native 已存在時，用 reader 核對既有的 ID 與資料，只補缺的部分，不用 `replace=True` 覆寫。
不手造 ledger 事件；Measure GUI 的 Load 清單直接掃描 `runs/*/data.h5`。Metadata 不可讀時，run_id 為空字串，experiment、point、時間與 completion 欄位留空。清單保留 exact Path 與錯誤，選取仍傳回 exact path，不從新舊目錄名猜身分。

完成條件：每個 Labber 檔都是 native 加原檔副本、都已讀回，或列 permanent-pending 並附缺少的證據。

## 9. 最後核對與刪除舊來源

在新的 process 裡讀回整個 entry：

```bash
uv run --directory "$REPO" --no-sync -- python - <<'PY'
import os
from zcu_lab.components import register_all
from zcu_tools.resources.entry import ResultEntry, component_registry
register_all(component_registry)
entry = ResultEntry.open(os.environ["NAME"], result_root=os.environ["RESULTS"],
                         database_root=os.environ["DATABASE"])
print("entry", entry.entry_id)
for label in entry.list_points():
    entry.use_point(label)
    print("point", label)
PY
```

預期印出原 UUID 與所有 points。程式能載入不等於內容正確，還要逐項看：

- [ ] 每個 `exps/<sub>` 都有 point，每張舊圖都在正確的 `figures/`，hash 相同。
- [ ] setup 與每個 point 的值都有來源；磁通數字都有同一份 meta 的單位，或標了「假設單位」。
- [ ] predictor、dispersive、t1_curve 格式檔齊全或已列 pending；每個 `null` 都有報告列。
- [ ] module 與 waveform 都已讀回或逐筆列 pending。
- [ ] Labber 都已轉 native 並讀回，或已列 pending；遷移賦值的 run_id 與 `$REC` 裡的對照一致。
- [ ] 報告的 `unfinished_publications`、`temporary_pending` 都是空的。只剩 permanent-pending，而且每筆都指到備份內的路徑。

再跑一次第 2 步最後的四個 `cmp`，確認備份與來源都沒變。來源出現新檔時停下並重新盤點，因為那可能是遷移期間新寫入的資料。

報告複製到 `$REC/manual-migration-report-<date>.json` 與 `$NEW_R/records/manual-migration-report-<date>.json`，不覆寫既有的報告。

使用者看過以上結果並授權刪除後，才執行：

```bash
realpath -- "$OLD_R" "$OLD_D" "$BR" "$BD" "$NEW_R"
realpath -m -- "$NEW_D"
rm -ri -- "$OLD_R" "$OLD_D"
test ! -e "$OLD_R" && test ! -e "$OLD_D"
```

只刪本次的兩個舊來源；其他 qubit、上層的 `<chip>/` 與 Database 根都留著。刪除後重讀新 entry，把刪除日期與範圍補進報告。

## 部分接手

適用於應急工具或人工只遷了一半的情況。先做第 1 到第 3 步，再保存現有新 entry 的完整副本與 hash 清單到 `$REC/existing-entry/`，之後才修改。
不重新 create entry，也不靠應急工具的 manifest 判斷成功。

以備份的盤點當左欄、現有新 entry 當右欄，逐項比對：

| 現況 | 做法 | 完成條件 |
| --- | --- | --- |
| 缺某些 points | 先補 setup 的缺值，再依第 6 步建缺的 point。 | 每個舊 label 對應一個 point，既有 point 不重建。 |
| point 已被手動改過 | 逐欄比較舊 meta 與目前值，只補缺值。 | 不同值列兩來源並 pending，由使用者決定是否覆寫。 |
| 圖在應急工具的 `migration-preserved/result/exps/<sub>/image/`，沒掛到 point | point 建好後，從備份用 `copy_checked` 複製到 `points/<sub>/figures/`。 | 原檔名與 hash 相同。preserved 那份可以保留。 |
| 只複製了 `params.json`，缺格式檔 | 依第 5 步寫 predictor、dispersive、t1_curve；`params.json` 副本從新 entry 移除。 | 格式檔齊全或已列 pending。 |
| 格式檔已存在或只寫一半 | 對照備份逐欄核對，只補缺項。 | 已有的不同值保留並列衝突。 |
| fit 被寫進了 setup 或 point 的 component 欄位 | 另寫 entry predictor；只移除能證明是 fit 誤寫的欄位。 | meta_info 本身的正式欄位保留。 |
| point 有 flux 數字卻缺單位 | 依第 4 步的單位規則補單位或標「假設單位」。 | 沒有單位證據也沒有宣告時，移除數字並列 pending。 |
| library 只換了 header，或有 dangling slot | 依第 6 步補轉換，有效的 Spec 不動。 | Spec、child key、slot 都核對過。 |
| processed spectrum 只做了原樣複製 | 保存現有檔與 hash，依第 7 步逐 group 建正式輸出與 residual companion，不在原檔上覆寫。 | 正式檔經 load_spectrums 讀回；未轉 datasets／attrs 在 companion 可見；已轉 data 不重複。 |
| Labber 已搬走，但沒有 native | 用原始 hash 清單確認備份裡有原 bytes，再依第 8 步轉換。 | 「已搬」與「已轉」分開記錄。 |
| 已有 native 或已固定 run_id | 用 reader 核對，沿用既有 ID。 | 有效的 native 不覆寫，固定的 ID 不重新分配。 |
| 來源與目的都在，但 hash 不同 | 兩份都保存，不看 mtime 選新的。 | 列衝突，使用者決定前不刪來源。 |

每補一項就讀回或 `cmp`，並更新該列的狀態。手動接手後，原工具可能無法再 resume；要切回工具前，先由工具 owner 確認接手方式。

## Entry 改名與後續保存

一般 entry 改名使用 `rename_entry(old, new, result_root=...)`，只移動 `result/<old>/` 到 `result/<new>/`，UUID 不變。Database 舊輸出不搬動，也沒有第二次目錄移動或復原步驟。以新名重新 open 後，新 Save 寫到 `Database/<new>/`，舊輸出留在 `Database/<old>/`。遷移未完成時先按固定名稱完成遷移，不在 resume 中途改名。

後續 Save 使用 `Database/<name>/runs/<experiment>_<run_id>/`，Result 圖片副本為 `<experiment>_<run_id>_<圖名>.png`。Workflow 的 Result／Database 兩側都用 `workflows/<workflow名>_<run_id>/`。這些路徑保留完整 run_id，實驗與 workflow 名中的 `/` 轉 `_`。

Labber 新 Save 仍用 `Labber/YYYY/MM/Data_MMDD/<experiment>_<id6>_<序號>.hdf5`。上述規則不重命名遷移保全的舊圖、workflow 或 Labber 檔，也不提供舊 run 目錄的相容邏輯，見 [ADR-0063](adr/0063-persistence-ownership.md#run-輸出位置與-writer-權責)。

## 把 results 搬到 result

他機寫在 `results/<name>/` 的 entry，只要整個目錄搬到 `result/<name>/`。這是單純的目錄搬移：UUID、point、內容與 `Database/<name>/` 都不變。
搬完不代表遷移已完成，原有的 pending 與清理條件照舊。

```bash
set -euo pipefail
export REPO=/absolute/path/to/Qubit-measure-gui NAME=migrated-Q0
export FROM="$REPO/results/$NAME" TO="$REPO/result/$NAME"
export BACKUP_ROOT="$(dirname "$REPO")/$(basename "$REPO")-backup"
export WORK="$BACKUP_ROOT/migration-records/root-move/$NAME"
hash_tree() { (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum); }
inventory_tree() { (cd "$1" && find . -printf '%y %m %P\0' | LC_ALL=C sort -z); }

test -d "$FROM" && test ! -e "$TO" && test ! -e "$WORK"
test -z "$(find "$FROM" -type l -o ! -type d ! -type f)"
mkdir -p -- "$WORK" "$(dirname "$TO")"
hash_tree "$FROM" > "$WORK/before.sha256"
inventory_tree "$FROM" > "$WORK/before.inventory"
cp -a -- "$FROM" "$WORK/entry-before-move"
cmp "$WORK/before.sha256" <(hash_tree "$WORK/entry-before-move")
mv -T --no-clobber -- "$FROM" "$TO"
test ! -e "$FROM"
cmp "$WORK/before.sha256" <(hash_tree "$TO")
cmp "$WORK/before.inventory" <(inventory_tree "$TO")
echo "Root move verified: $TO"
```

完成條件：最後印出 `Root move verified`，搬移前的副本留在 `$WORK`。
`$TO` 已存在時不要執行 `mv`，改走[部分接手](#部分接手)逐項比對。只搬 `results/<name>`；`results/` 裡可能還有其他 entry，不要整個刪掉。
搬完後用新路徑重新開啟 entry，核對 UUID 與 points。

## TestChip/Q0 核對基準

repo 範例的盤點結果：5 個 exps、14 張 PNG，label 是 `062823`、`070216`、`070716_0.500`、`070800_0.500`、`072523`。
`params.json` 只有 `schema_version`、`project`、`name`，沒有 `fluxdep_fit`，所以不會產生 predictor。
手動完成後應有 5 個 points、14 張 point figures，每張 hash 都一致。
