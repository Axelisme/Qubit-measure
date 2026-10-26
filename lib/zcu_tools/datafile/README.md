# zcu_tools.datafile

**Last updated:** 2026-10-27

**讀者：** framework-dev。讀寫通用 HDF5 容器，或獨立轉出 Labber。

## Container

`datafile` 只處理數值陣列、有限 JSON 與 opaque HDF5 子樹，不解釋實驗欄位。
公開名稱由 package root 按需載入。Path helpers 不要求 h5py。
格式與依賴規則見 [ADR-0080](../../../docs/adr/0080-datafile-generic-container.md)。
`datafile-standalone` import contract 禁止本 package import 其他 framework packages。

一份 `zcu.container` 3.0 檔案有三個命名空間。

| 命名空間 | 公開值型別 | 契約 |
|---|---|---|
| `vars/<name>` | `Variable(values, axes, unit)` | 整數、浮點或複數 NumPy 陣列。軸按 NumPy 維度順序排列，各軸長度等於對應維度。Scalar 沒有軸。 |
| `docs/<name>` | `JsonObject` | UTF-8 scalar dataset。只接受有限的 JSON object，不轉換 NumPy 值，不解釋鍵。 |
| `groups/<name>` | `GroupSource` | 原樣複製生產者的 HDF5 子樹，包括 attributes、links 與內部 references。 |

三個命名空間可以為空。名稱必須是非空路徑段，不能含 `/` 或 NUL，也不能是 `.` 或 `..`。
同一 variable 的軸名稱不能重複。不同命名空間可以重用名稱。
每個 var 的 `values` dataset 以 dimension scales 連到 `axes/<axis-name>`。
數值陣列可以包含 NaN；JSON 不能包含 NaN 或 Infinity。

`Container(vars, docs, groups, attrs)` 是讀寫共用的值型別。
`ContainerAttrs(format_version, producer, kind, extra)` 保存根屬性。
新檔由 caller 指定版本 `"3.0"` 與 producer，kind 可以是 None。
extra 只接受有限 JSON scalar。JSON null 以 HDF5 null-dataspace attribute 表示。
`format` 固定為 `"zcu.container"`，不能透過 extra 覆寫保留的 header keys。

```python
from pathlib import Path
import numpy as np
from zcu_tools.datafile import Axis, Container, ContainerAttrs, Variable, load, save

container = Container(
    vars={"voltage": Variable(np.array([1.0, 2.0]), (Axis("time", "s", np.array([0.0, 1.0])),), "V")},
    docs={"note": {"operator": "example"}},
    groups={},
    attrs=ContainerAttrs("3.0", "example-producer", None, {}),
)
save(Path("data.h5"), container)
loaded = load(Path("data.h5"))
save(Path("copy.h5"), loaded)
```

`save(path, container, *, replace=False)` 寫 caller 的 exact path。
預設拒絕既有檔案。它先寫同目錄 temporary，關檔後才發布。
`replace=True` 使用 atomic replace，失敗保留原目的檔並清理 temporary。
這不是跨檔交易，也不承諾掉電恢復。

`load(path)` 關檔後回傳 detached arrays、docs 與 groups。
載入的 GroupSource 持有 detached file image，原檔刪除或改名後仍能寫回。
Caller 也可以用 `GroupSource(existing_path, "/subtree")` 提供自己的子樹。
若改變 vars 的形狀，groups 指向該 var 的 HDF5 references 不保證仍有效。

`read_doc(path, name)` 回傳一份 detached `JsonObject`，只讀 header 與指定文件。
它不讀 vars、其他 docs 或 groups。壞 vars 不阻礙合法文件讀取。
名稱遵守命名空間的路徑段限制。缺文件 raise `KeyError`，錯誤包含來源檔與文件名。
格式或文件非法 raise 指名來源檔的 `FormatError`，I/O 錯誤原樣傳遞。
此入口與 load 都只接受 zcu.container 3.x。
讀到合法文件不代表整份容器可由 load 成功載入。

`inspect(path)` 回傳 `ContainerInfo`，只讀 header、var schema 與 doc/group 名稱。
它不載入數值或座標陣列，回傳 axes 的 values 為空陣列。
它不解碼 docs，也不保證整份檔案可由 load 成功載入。
格式錯誤回 `FormatError` 並標示來源路徑；檔案 I/O 錯誤原樣傳遞。

## Versions

Reader 接受 major 3 的任何 minor。`load` 回傳來源版本，`save` 原樣保留，不降 minor。
未知的 vars、docs、groups 與 extra 根屬性沿 `load`、修改、`save` 保留。
格式名稱錯誤或 major 不同時，reader 拒絕載入。

舊 `zcu.experiment-data` 2.x 不在支援範圍，reader 直接拒絕。

cfg、metadata、analysis、source 都是生產者的文件命名約定，不是容器的保留槽位。
Experiment 擁有 run 型別與 cfg 驗證。Artifacts 擁有 analysis JSON 信封，見
[experiment](../experiment/README.md)。

## Labber

Labber codec 與容器核心分開。`write_labber(path, container, *, metadata=None, grouped_layout="common")`
只消費 vars，不解釋 docs 或 groups。Caller 以 `LabberMetadata` 明傳 comment、tags、project、user、creation_time。
需要 cfg comment envelope 時，由 caller 使用 `encode_labber_comment(values, comment, snapshot=...)`。
Snapshot 是可選 JSON object；experiment saver 傳入歷史 point 與 roles，與 cfg 並列。
Codec 只驗有限 JSON，不解讀 snapshot keys，不改 Container 或 wire version。

Labber 至少需要一個 real step axis。單一 var 寫 single log，多個 vars 預設要求 common grid。
明示 `grouped_layout="independent"` 時，各 rank-1/2 var 寫自己的 root／Log_N。
內部 helper 直接寫 marker-qualified grouped v1，並標 finalized，不經中介容器檔案。

`read_labber(path)` 回 `(Container, LabberMetadata)`，保留 NumPy 軸順序。
`groups/labber` 保存原始 Labber 子樹與 timestamps。
需要直接讀 per-trace timestamps 時，可用 `load_labber_data`、`load_labber_logs` 或 `load_grouped_labber_data`。
這些低層讀寫能力保留 Labber 的 inner-first axes 與完整 metadata，不推測實驗語意或做單位轉換。
歷史 Labber Instrument config 與 comment 解碼只在 `migrations/storage_migration/`。

`paths` helpers 只處理 suffix 與 caller 明示的檔名規則。
`reserve_labber_filepath` 計算當時未使用的序號路徑，不提供跨程序 reservation。
保存位置、batch、ledger 與失敗隔離由 caller 擁有，見 [ADR-0063](../../../docs/adr/0063-persistence-ownership.md)。
