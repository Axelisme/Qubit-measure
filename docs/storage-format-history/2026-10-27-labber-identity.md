**讀者：** framework-dev、agent。辨識 2026-10-27 前後的 Labber 身分信封與存檔命名，判斷是否需要遷移。

# Labber 目標身分與 caller 命名

日期為 2026-10-27。範圍是 GUI、Notebook run 保存與 workflow Labber export 的 comment，以及 Labber 檔名與單次 native run 目錄名。Container 內容格式、ledger、component record table、workflow 雙側 root 與 raw operation 檔名不變。

## 改前形狀

Labber comment 只有 cfg、可選使用者備註與格式化時間。例如：

```json
{
  "cfg": {"frequency": 5000.0},
  "comment": "sweet spot",
  "timestamp": "2026-10-27 12:00:00"
}
```

它沒有 point 或 roles。單 qubit cfg 通常只記硬體頻道、頻率與增益，檔案離開 entry 後無法從內容取得 qubit／resonator 名稱。

啟動名稱中的斜線轉成底線。Labber 用 run_id 後六碼及從 1 開始的序號，native 用完整 run_id。例如同一 run_id `20261027T120000Z-a1b2c3` 的路徑為：

```text
Labber/2026/10/Data_1027/twotone_time_domain_t1_a1b2c3_1.hdf5
runs/twotone_time_domain_t1_20261027T120000Z-a1b2c3/data.h5
```

Native 的 docs/metadata.snapshot 已有 entry_id、point 與 roles，這次不改它。

## 改後形狀

上述 Labber comment 在 cfg 旁新增 snapshot，保留原備註及 timestamp：

```json
{
  "cfg": {"frequency": 5000.0},
  "snapshot": {
    "point": "Q1_operating_point",
    "roles": {"qubit": "Q1", "resonator": "R1"}
  },
  "comment": "sweet spot",
  "timestamp": "2026-10-27 12:00:00"
}
```

GUI Adapter.save_stem 委派 adapter 自己 spec 上的 callback。Notebook 沿 core spec 的相同欄位，workflow export 由 caller 傳 callback。三者都收到 SaveContext，含前端完整名稱、完整 run_id、歷史 point、解析後 roles、可用 cfg values 與 saved_at。框架使用回傳的 stem，不另補前綴或 run_id。

預設 stem 將名稱中的 `/` 轉 `.`，接 role values 與完整 run_id，各段用 `_` 分隔。Role values 按 mapping 順序以 `-` 連接，清單按原順序攤平，空值省略。沒有角色時不產生角色段。相同範例現在寫成：

```text
Labber/2026/10/Data_1027/twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3.hdf5
runs/twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3/data.h5
```

Mux callback 依 cfg.participants 的原始索引順序列 qubits，例如 `mux.t1_Q1+Q2+Q3_20261027T120000Z-a1b2c3`。Readout-only onetone 沒有完整 qubit 身分，因此列 readout labels，例如 `mux.onetone_R1+R2_20261027T120000Z-a1b2c3`。缺 cfg 或名字時使用預設 roles 命名，不讀 live context 補值。

Labber 初始名字已存在時，仍沿用 numeric suffix helper。例如上述初始檔已存在，下次選 `twotone.time_domain.t1_Q1-R1_20261027T120000Z-a1b2c3_1.hdf5`。Helper 不作跨程序原子保留。Native 同名仍由 saver 以 replace=True 原子覆寫，不加避讓。自訂 callback 若每次回傳不同 stem，native 保存目錄也會跟著變。

## 為什麼改

檔名只有實驗名稱與 id6，無法讓操作者分辨 Q1、Q2 或 resonator。單 qubit Labber comment 又缺少名稱，移出 entry 後就失去目標身分。現在 comment 帶歷史身分，預設檔名也讓人直接辨識目標。

實驗作者決定哪些資訊進檔名，框架只提供組名所需的事實。使用者於 2026-10-27 裁決：「Q-L3我的想法是adapter決定怎麼生成名字，可以有預設實現。」並要求：「run_id也是由adapter行數補上，請你遵守框架不做多餘行為的設計理念，將可能需要的資訊作為參數傳給save_stem。」因此自訂 callback 的結果可以完全不含 run_id。

2026-10-27 合併前檢查發現，初版預設只取 id6 會讓不同 run 共用 native stem。Writer 的 replace=True 因而可能覆寫另一份 acquisition，而不只是重存同一 run。預設與 mux callback 因此保留完整 run_id。框架仍不改寫 callback 的結果，自訂 callback 必須避免不同 run 共用 stem。

Results browser 不再靠目錄的完整 run_id 尾碼識別 figure-only run。它先讀 native metadata；沒有可讀 metadata 時，用 saved event 的成功 output exact path 找 acquired 事實。沒有這些證據時保留 unknown，不猜 id6。

## 如何辨識

先讀 Labber 根屬性的 comment JSON。若有 snapshot.point 與 snapshot.roles，就是包含這次身分擴充的信封。沒有 snapshot 的歷史檔仍可讀，但不具有這份可攜目標證據。

預設檔名中的 dotted 名稱與 role values 可作提示，不能用來判版。自訂 stem 可以採任意安全名稱，碰撞 suffix 也不是版本。完整 run_id 仍以 native docs/metadata 或 ledger 為準，不從 stem 還原。

Native 根屬性仍是 zcu.container 3.0，docs/cfg 與 docs/metadata 的形狀不變。單看 native 格式版本無法判斷這次命名規則，應檢查保存路徑及相關輸出證據。

## 遷移

不遷移，不重命名舊輸出。歷史 Labber 沿既有 reader 讀取，沒有 snapshot 時不補造。讀 comment 的 consumers 忽略不支援的 keys，新增 snapshot 不要求升 wire 或 Container 版本。

離線 storage migration 明確傳原有 experiment 加完整 run_id 的 native stem，因此既有 migration destination 與 resume 驗證不變。新規則只影響這次起由 runtime saver 產生的路徑。

## 相關 ADR

- [ADR-0063](../adr/0063-persistence-ownership.md) 擁有保存權威、caller 命名、snapshot 與碰撞邊界。
- [ADR-0080](../adr/0080-datafile-generic-container.md) 擁有通用容器與獨立 Labber codec，這次不改其格式版本。
