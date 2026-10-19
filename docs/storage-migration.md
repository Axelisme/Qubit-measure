# 舊資料遷移指南（手動）

本指南把一個舊 qubit 的資料，也就是 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`，整理成一個新格式 entry。
全程用 shell、編輯器與 repo 的公開 API 完成，不需要遷移腳本。

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
│   │   ├── points/<label>/{point.yaml,module_cfg.yaml,figures/}
│   │   └── records/manual-migration-report-<date>.json
│   └── Database/<name>/
│       ├── runs/<run_id>/data.h5         ← Labber 轉成的 native
│       ├── Labber/<原相對路徑>            ← 原 Labber 檔，bytes 不變
│       ├── arb_waveforms/**
│       └── migration-preserved/**        ← 無法轉換、需保全的檔
└── Qubit-measure-gui-backup/             ← 平行備份根：舊格式，路徑與 repo 內相同
    ├── result/<chip>/<qub>/              ← 原樣複製
    ├── Database/<chip>/<qub>/            ← 原樣複製
    └── migration-records/<chip>/<qub>/   ← hash 清單、工作紀錄、報告
```

三條規則決定整個流程：

1. **平行備份。** 舊來源整份複製到 repo 旁的 `<repo 目錄名>-backup/`，保留 repo 內的相對路徑，逐檔核對 hash。每次遷移都累積到同一個備份根。
2. **新格式寫回原 repo。** 新 entry 直接寫在同一個 `result/`、`Database/` 根下，與舊來源不重疊。
3. **最後刪舊來源。** 新 entry 全部核對完、使用者授權後，刪掉 repo 內本次的 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`。repo 從此只剩新格式；舊格式只存在平行備份裡。

## 選路線

| 現況 | 走法 |
| --- | --- |
| 舊資料還沒動過 | 依序做第 1 到第 9 步。 |
| 已有部分新 entry（應急工具或人工只做一半） | 做第 1 到第 3 步，再到[部分接手](#部分接手)。 |
| 他機把新 entry 寫在 `results/<name>/`（多一個 s） | 只需[搬到 result](#把-results-搬到-result)，不重做遷移。 |

## 名詞

- **來源**：本次的 `result/<chip>/<qub>/` 與 `Database/<chip>/<qub>/`，包括隱藏檔、空目錄與未知檔。
- **備份**：平行備份根裡與來源相同相對路徑的副本，hash 與來源逐檔相同。後續一律從備份讀，來源保持不動直到第 9 步。
- **報告**：`manual-report.json`，每個來源檔與每個未轉欄位一列，記去向與狀態。它是交接清單，不是工具的 resume manifest。
- **狀態**：`verified` 表示目的已讀回核對；`temporary-pending` 表示還能完成，只是尚未完成；`permanent-pending` 表示缺證據無法轉換，列出備份內的位置與缺的證據。
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
- `BACKUP_ROOT` 在 repo 外。`BR`、`BD`、`REC` 都還不存在；已存在表示這個 qubit 遷移過，改走[部分接手](#部分接手)。
- 全新遷移時 `NEW_R`、`NEW_D` 都不存在；已存在就改走部分接手，不刪掉重來。
- 新 entry 不包住、也不透過 symlink 指進任一來源。
- `find "$OLD_R" "$OLD_D" -type l -o ! -type d ! -type f` 沒有輸出。有輸出就停下，向使用者確認實體來源。
- `OLD_D` 不存在時，記下並取得使用者確認；之後只備份 `OLD_R`，不建立空 Database 充數。

## 2. 備份到平行備份根並驗證

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
| 根層 `params.json` 的 `fluxdep_fit` | `predictors/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的 `dispersive`，以及 `fluxdep_fit.plot_transitions` | `dispersive/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的 `t1_curve_fit` | `t1_curve/<Q>.json` | 第 5 步 |
| 根層 `params.json` 的每個區塊原文 | `components.<Q>.ext.legacy_params.<區塊名>` | 第 4 步。`params.json` 本身不複製進新 entry。 |
| 根層 `module_cfg.yaml` | entry 的 setup library | 第 6 步 |
| `exps/<sub>/meta_info.json` | `points/<sub>/point.yaml` | 第 6 步 |
| `exps/<sub>/module_cfg.yaml` | `points/<sub>/module_cfg.yaml` | 第 6 步 |
| `exps/<sub>/image/*.png` | `points/<sub>/figures/<原檔名>` | 第 7 步 |
| 舊直屬 `<sub>/{meta_info.json,module_cfg.yaml}` | 同名 point | 與 `exps/<sub>` 同名時停下，列出兩個來源，不合併。 |
| exps 以外的圖 | `result/<name>/figures/<原相對路徑>` | 第 7 步 |
| `samples.csv` | `samples/<Q>.jsonl` | 第 7 步 |
| `arb_waveforms/**`（result 或 Database） | `Database/<name>/arb_waveforms/**` | 第 7 步 |
| `autofluxdep_runs/**`、其他未知 result 檔 | `Database/<name>/migration-preserved/result/<原相對路徑>` | 第 7 步，原樣保全。 |
| Database 的 Labber 檔 | `Database/<name>/runs/<run_id>/data.h5` ＋ `Database/<name>/Labber/<原相對路徑>` | 第 8 步 |
| 其他 Database 檔 | 留在備份；需要副本時放 `migration-preserved/Database/<原相對路徑>` | 報告記原位置與未知用途。 |

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

印出的 UUID 與兩個路徑記入 `$REC/notes.md`。部分接手時跳過這段，沿用既有 UUID。

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
   未列出的 key 原文保全並列 pending；歷史 key 全表在 [storage_mapping.py](../migrations/storage_mapping.py)。

2. **單位照抄，不換算。** 頻率 MHz、時間 us、能量 GHz。`flux_unit` 管 `cur_A` 與 `flx_*`，`jpa_flux_unit` 管 `best_jpa_flux`。磁通數字依序取單位：
   1. 同一份 meta 明寫的單位。
   2. entry 已知的明寫單位：根層 meta_info 的單位；沒有時用 `params.json` 根層的 `flux_unit`。
   3. 使用者宣告的假設單位（本專案的資料用 `mA`），報告標「假設單位」。
   4. 以上都沒有：數字不寫入，列 pending。

   明寫的單位互相衝突時列 pending，不擇一。報告記下實際採用的是哪份檔的單位。

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
- `source.created_at`：只有原區塊 `timestamp` 是含時區的 ISO 8601 時才照抄原字串；否則 `null`，報告列 pending，附備份內原文位置。entry UUID、mtime 與遷移時間都不能代填。
- 未知數值填 `null` 或省略，不補 0。已知數值必須是有限數字，不接受 bool 或數字字串。
- 格式檔只放在 entry 層，不複製到 points。原區塊仍在 `legacy_params` 與備份裡。

寫完每個檔都要用 JSON parser 讀回一次。格式 owner 的 Python 模組 land 後，改用它們的 `load_*`、`save_*` 讀寫與驗證，見 [resources README](../lib/zcu_tools/resources/README.md)。

### predictor：`predictors/<Q>.json`

只轉舊 Fluxonium fit（`fluxdep_fit.params` 含 `EJ`、`EC`、`EL`）。transmon 或已帶 `model` 的 fit 列 permanent-pending。

| 舊值 | predictor 欄位 |
| --- | --- |
| `fluxdep_fit.params.EJ`、`EC`、`EL` | `model: "fluxonium"`，`params.EJ`、`params.EC`、`params.EL`，GHz |
| `fluxdep_fit.flux_half`、`flux_period` | `params.flux_half`、`params.flux_period`，不換算 |
| （舊 predictor 的預設） | `params.flux_bias: 0.0` |
| `fluxdep_fit.flux_int` | 不寫入；它只用於舊 alignment，原文留在 `legacy_params` |
| params 根層 `flux_unit` 或 `fluxdep_fit.flux_unit` | `flux_unit`。只有一處時用該值；兩處相同時用該值；兩處衝突時寫 `null` 並列 pending；兩處都沒有時用使用者宣告的假設單位並標「假設單位」，沒有宣告就寫 `null` 並列 pending |
| `fluxdep_fit.timestamp` | `source.created_at`，規則見上 |

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
只有 `plot_transitions`、沒有 `dispersive` 區塊時也要產檔，`g`、`bare_rf` 寫 `null`。
`g`、`bare_rf`、`plot_transitions.r_f` 的單位是 GHz；`plot_transitions` 其他欄位原文照抄。
`source.created_at` 取 `dispersive.timestamp`；只有 `plot_transitions` 時取 `fluxdep_fit.timestamp`。

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

1. 刪除 `general.entry_id`，因為目前的 point loader 不接受這個欄位。setup 的 UUID 保持不變。
2. `general.created_at` 改成剛印出的時間，也就是新 point 的建立時間。
3. 套用 `exps/<sub>/meta_info.json`：用第 4 步的對照表，point meta 有的值覆蓋 setup 繼承的值。
4. 單位依第 4 步的順序決定。point meta 沒寫單位時，沿用從 setup 繼承的單位，報告記「unit from root meta_info.json」。
   point meta 寫了與 setup 不同的單位時，移除從 setup 繼承的 `flx_half`、`flx_int`、`flx_period` 及其 metadata，只留 point meta 自己的數字，不讓 setup 的數字配上 point 的單位。

完成條件：每個 `exps/<sub>` 對應一個 point，各自有完整 components、metadata 與正確的 flux 單位。
`entry.list_points()` 只列出 `point.yaml`、`module_cfg.yaml` 兩檔都在的目錄，所以另外用 `find "$NEW_R/points" -mindepth 1 -maxdepth 2` 檢查未完成的骨架。

### module_cfg：轉成正式 Spec

舊 `module_cfg.yaml` 的 `modules`、`waveforms` mapping 不是新 library，只換 header 不算轉換。逐筆建立「舊 family、舊 key、新 canonical key、引用位置」的對照，轉成 `components.<Q>.{pulses,readout,reset,waveforms}.<name>`，再把 point 的 module slot 指到新 key，例如 `components.Q0.module.x180: Q0.pulses.pi_amp`。

1. 用 [cfg_model](../lib/zcu_tools/cfg_model/README.md) 的 `module_spec_from_raw(raw)`、`waveform_spec_from_raw(raw)` 驗證每筆候選；child key 用 `remap_module_references`、`remap_waveform_references` 改。expression 不求值，缺的必要欄位不猜。
2. readout 要轉成平鋪的 `readout/pulse` Spec：generator 是 `gen_ch`，ADC 是 `ro_ch`，只有一個 `freq`。不能確定舊 `pulse_cfg`、`ro_cfg` 怎麼組合時，該筆列 pending。
3. 用 owner 提交並讀回：

```python
from zcu_tools.cfg_model import waveform_spec_from_raw
spec = waveform_spec_from_raw(raw_waveform)
with entry.module_library(label).edit() as draft:
    draft.set_waveform("Q0.waveforms.drive", spec)
print(entry.module_library(label).snapshot().waveform_spec("Q0.waveforms.drive"))
```

`entry` 依第 9 步的方式註冊 component models 後開啟。provenance 由 owner 產生，不手寫事件或 run_id。
一般讀取 library 時會自動備份並移除壞掉的條目，所以先在工作副本試轉候選，不拿真檔試錯。

**根層 `module_cfg.yaml`** 用同樣規則轉成 entry 的 setup library，新 point 從 setup library 繼承。目前 main 的 `module_library` 只接受 point label；setup library 的支援 land 前，根層 module_cfg 列 temporary-pending，land 後再補。

完成條件：每筆舊 module 和 waveform 都已轉換並讀回，或逐筆列 pending。空 library 本身合法，但不能說成「舊 library 已全部轉換」。

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
- **`autofluxdep_runs/**` 與未知檔**：複製到 `$NEW_D/migration-preserved/result/<相對路徑>`，報告註明它不是新格式資料。
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

  回傳的 pending 列逐列記入報告。使用者沒有指定值欄或單位時，整份 CSV 列 pending。用法細節見 [migrations README](../migrations/README.md#sample-csv--jsonl)。

完成條件：每對 hash 相同；每張 `exps` 圖都掛在正確的 point。helper 中斷可能留下半份目的檔，記入 `unfinished_publications`，從備份核對後再補。

## 8. Labber 檔轉 native

Labber 與 native 都是 HDF5，但格式不同，把 `.hdf5` 改名成 `data.h5` 不會得到 native。

**必要證據**齊全才轉：experiment tag 與 cfg_type、完整歷史 cfg、variable、axes 與 signal 的名稱、單位、dtype 和 shape。缺任何一項就整檔列 permanent-pending，原檔留在備份裡。

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

以下是合成範例。`synthetic_demo`、`DemoCfg` 不是 repo 已註冊的實驗；真檔要換成確認過的 `(tag, cfg_type, schema_version)`、完整歷史 cfg 與正式 schema。

```python
import hashlib, json, os
from pathlib import Path
from uuid import UUID
import numpy as np
from migrations.legacy_labber import load_legacy_labber_payload
from zcu_tools.datafile import (
    AxisSchema, CfgSnapshot, DataVariable, RunMetadata, RunSnapshot,
    SoftwareProvenance, VariableSchema, load_run_data, save_run_data,
    validate_experiment_payload,
)
from zcu_tools.resources.entry import new_run_id

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
destination = Path(os.environ["NEW_D"]) / "runs" / metadata.run_id / "data.h5"
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
不手造 ledger 事件；Measure GUI 的 Load 清單直接掃描 `runs/*/data.h5`。

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
realpath -- "$OLD_R" "$OLD_D" "$BR" "$BD" "$NEW_R" "$NEW_D"
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
| 圖在 `migration-preserved/result/exps/<sub>/image/`，沒掛到 point | point 建好後，從備份用 `copy_checked` 複製到 `points/<sub>/figures/`。 | 原檔名與 hash 相同。preserved 那份可以保留。 |
| 只複製了 `params.json`，缺格式檔 | 依第 5 步寫 predictor、dispersive、t1_curve；`params.json` 副本從新 entry 移除。 | 格式檔齊全或已列 pending。 |
| 格式檔已存在或只寫一半 | 對照備份逐欄核對，只補缺項。 | 已有的不同值保留並列衝突。 |
| fit 被寫進了 setup 或 point 的 component 欄位 | 另寫 entry predictor；只移除能證明是 fit 誤寫的欄位。 | meta_info 本身的正式欄位保留。 |
| point 有 flux 數字卻缺單位 | 依第 4 步的單位規則補單位或標「假設單位」。 | 沒有單位證據也沒有宣告時，移除數字並列 pending。 |
| library 只換了 header，或有 dangling slot | 依第 6 步補轉換，有效的 Spec 不動。 | Spec、child key、slot 都核對過。 |
| Labber 已搬走，但沒有 native | 用原始 hash 清單確認備份裡有原 bytes，再依第 8 步轉換。 | 「已搬」與「已轉」分開記錄。 |
| 已有 native 或已固定 run_id | 用 reader 核對，沿用既有 ID。 | 有效的 native 不覆寫，固定的 ID 不重新分配。 |
| 來源與目的都在，但 hash 不同 | 兩份都保存，不看 mtime 選新的。 | 列衝突，使用者決定前不刪來源。 |

每補一項就讀回或 `cmp`，並更新該列的狀態。手動接手後，原工具可能無法再 resume；要切回工具前，先由工具 owner 確認接手方式。

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
