# 0082 — Component flux frame 與明確升級

日期：2026-10-27
狀態：Accepted；既有名稱與 membership 接線的 advisor 細節仍為 Review: pending。
相關：[[0063]]、[[0069]]、[[0072]]、[[0081]]

## 脈絡

一顆 qubit 可能同時受到 local 線、global coil 與其他 DC 線影響。
若每條線的 integer 都以其他線停在各自 integer 為條件，第一條線就無法獨立校準。
只保存儀器值的 record，也無法在 flux jump 或重新校準後還原當時的 flux。

Component schema 與 YAML 容器各自演進。
讀取時猜格式會把相容邏輯分散到每個 caller，因此舊文件經明確命令升級。

## 決策

### Component 保存 offset 與 signed periods

每顆 qubit 保存 `flux: {offset, period: {<device>: ...}}`。
未校準形狀為 `{offset: null, period: {}}`。
Offset 是所有儀器輸出 0 時的 reduced flux，以 Φ0 表示，不做 mod 1。
每台儀器只有一個有限、非零的 signed period，單位為該儀器原生單位／Φ0。
負 period 表示增加 bias 會減少 flux。

```text
phi = offset + sum(value[d] / period[d])
```

沒有 period 的儀器不計入。有 period 但當下沒有值的儀器也不貢獻，period 原樣保留。
明寫無效值仍報錯。缺席規則適用於 writeback、record、分析與 form。
Half／integer 偏壓由模型投影取得，不再存成每台儀器的參考點。

Qubit、mux 與 JPA 的 `flux_dev` 指定主要／預設控制儀器，不參與公式，也不必有 period。
Mux group 不另存 qubit 校準。`coil` 不是保留別名。
Form 從固定 capture 的 `flux_dev` 選預設儀器，使用 detached device 值投影 half／integer。
缺 device、offset 或選定 period 時，對應必填初值留空，不猜數字。
Mux local 使用各 qubit 的 `flux_dev`，global 使用 group 的 `flux_dev`。

使用者於 2026-10-27 提出：「当所有仪器都是零的情况下，本身的 flux bias 单位是 flux quanta 本身，然后每一个仪器就只有。将其值转换成 flux quanta 的一个转换系数，就不用有锚定点了。」
對預設控制儀器的角色確認：「我傾向方案1，我可以理解成main_dev嗎？也就是這個原件的flux主要或是預設控制儀器？」

名稱與註冊共用 `cfg_model.validate_device_name`，拒絕 ASCII 點號、空字串與全空白，其餘原樣保留。
Device membership 按使用處的資料檢查，不新增 setup catalog。
拒絕空白名稱、detached device 選項與 membership 接線為既有 advisor 細節，Review: pending。

### 量測 fluxdep writeback 與 app Export params 分開

量測掃儀器 s 後，writeback 分開提出 `<qubit>.flux.period.<device>` 與 `<qubit>.flux.offset`。
Period 更新大小、保留原方向，首次校準使用正方向。Offset 項預設勾選，由 operator 決定是否接受。

```text
offset = 0.5 - half_fit / signed_new_period[s] - sum_other(value[o] / period[o])
```

其他儀器值取 Run-start snapshot。被掃儀器只取擬合 half，覆蓋 snapshot 的舊值一次，不重複計入。
Note 列出其他有 period 且有 snapshot 值的儀器，包含值為 0 者，寫出名稱、值、原生單位與扣除貢獻。
缺席儀器不貢獻，也不列 note；沒有其他儀器時 note 為 None。
不因同 Run 同步掃描其他線拒絕 proposal，實驗作者負責其實驗含義。
缺整份 Run snapshot 仍無法換算，不以 live device 補值。

同次 ordinary record 保存原始 half、integer、正的 period 與被掃 device。
Devices 來自完整 Run-start snapshot，flux_frame 來自該 Run 的固定 parameter capture。
接受時不重讀目前 point、alias 或 live device，也不把新校準放回該列 frame。

獨立 fluxdep app 的 Export params 只更新 component EJ／EC／EL 與選定 `flux_dev` 的 period。
單儀器、多儀器與 offset=null 都保留 component offset。缺 `flux_dev` 時跳過 period。
Receipt 維持 `written`／`skipped`，不新增 `partial`，也不把它改為 offset-only receipt。
資料可能經歷 flux jump，單筆 spectrum 的對齊不代表目前 component offset。
使用者於 2026-10-27 指出：「多次twotone/onetone數據間可能經歷幾次的flux jump，所以往往會有幾次看起來不一致的flux position校準，不可能變成一個跨數據都合理的flux參數設置」。

### Record 與分析

Lab Record Policy 要求 T1、T2、frequency、fluxdep 與 workflow records 保存同形狀的 `flux_frame`。
Notebook 作者量測前捕捉 frame，經既有 `SavedRun.accept(records=...)` 明傳。
`record_flux_frame` 配對固定 frame 與同次 Run snapshot，缺儀器值不貢獻，無效值仍指名報錯。
整個 snapshot 缺席時，一般 GUI adapter 省略 record，workflow 保留 `devices=null`。
Framework append owner 加 `component` 信封，不定義 flux 物理意義。

`flux_from_frame` 逐列依 explicit flux、row frame、caller fallback 的順序換算。
Row frame 使用 `flux_frame.offset` 與 `flux_frame.period.<device>`。
缺 offset 報錯並指出列與欄位，不退回 fallback。
Nested 缺 device value key 不貢獻，明寫 NaN、inf、bool 報錯。
Dotted 的 device value 或 period NaN cell 視為缺席，inf 與 bool 仍報錯。
攤平無法區分缺 key 與明寫 NaN，只有 nested 保證拒絕明寫 NaN，不新增 presence 欄位。

Fallback 統一使用 `analysis.flux.FluxFrame`，有值儀器的單位須與 caller 宣告相同。
獨立 `resolve_record_flux`／`validate_record_table_v2` 保留 `dev_value`／`dev_unit` 與歷史 flat row schema。
它們的 fallback 也使用新 `FluxFrame`，但必須恰有一台儀器，否則報錯並說明台數。
不保留舊 integer fallback 型別或相容讀取。
歷史改名的 records 保留當時名稱與 frame，不跟著改 component。

### Component 文件版本與升級

Setup、point 與 entry template 共用整份文件的 `schema_version`，runtime 只接受 3。
缺版或不同版本時列出文件版本、目前版本與長期 upgrade 命令，讀取不升級、不改檔。
`resources.parameter_upgrade` 擁有版本鏈、dry-run、備份與寫入。Lab 擁有分類、逐版驗證與純函式步驟。
未標版文件只有在明確 upgrade 且完整分類成功後才視為 v1。
Partial／corrupt 不寫入，其餘文件繼續。Apply 先備份原 bytes，current 文件驗證後跳過。
歷史步驟只在使用者確認資料都已升版後刪除。

保留 v1→v2→v3。V2 仍是每台儀器 `{half, integer, period}` 與 `flux_source`，以其他儀器停在各自 integer 為參考點。
V1 自己的線保留原轉換。Coil 只保留 signed period，在 v2 寫 integer 0、half = period／2，原 half／int 列報告。
Native 依自己的線、指定 coil、唯一 coil 選來源，多 coil 未指名則不選，保留全部 periods 並提醒重校。

V2 主儀器為 `flux_source`。有對應 entry 時：

```text
offset = 0.5 - half_main / period_main - sum_other(integer[o] / period[o])
```

不做 mod 1，保留主要儀器 half 分支。沒有來源或來源缺 entry 時 offset=null，所有 signed periods 保留。
空 flux 變成未校準形狀。此步才將 qubit、mux、JPA 的 `flux_source` 改名 `flux_dev`。
缺名用不同的 `unnamed_device`／`unnamed_coil`，兩個 CLI 逐實際目的位置提醒改名。
不同來源候選 offset 不一致可能是 flux jump，不歸腐敗，也不要求一致或任選第一筆。

Storage 在完整 seed／value batch 組成 EditView 前轉換，不先發布舊文件。
每批與最終的 `flux_dev` 都只來自明示來源，包含完整 overlay 的有效 setup／mapping 繼承來源，否則留空。
自動佔位不成為來源。Offset 每批按 v1 結構角色重算，自己的線優先，其次唯一 coil，否則 null。
經 v2 再升 v3 時也由 v1 結構角色提供主線身分，不從佔位名稱猜測。
完整有效且有明示來源的 point 校準取代繼承 offset 與該儀器 period，保留其他 periods。
原資料明示的 `unnamed_*` 照明示處理，不全域排除字串，不新增 provenance API 或 finalizer。
同批撞鍵、partial、非法名稱、數值與 spacing 仍拒絕，native 仍拒絕混版。
長期鏈不依賴 `migrations/`，CLI 為 `scripts/upgrade_parameters.py`。
Record 範圍只按 operator 宣告轉換舊無名值，不補歷史 frame 或 component。

## 邊界與可執行契約

沿用 [[0072]] 的 lab→framework 方向。Flux 數學只收純 analysis 型別，不讀 component。
Lab 負責 component 轉換、表單初值與 Measure proposal，三個數學入口統一處理正向、投影與反解。
Workflow FrequencyModel 的欄位與單儀器 affine 行為不變，初值由 `flux_dev` 投影 half 與 signed period。
Predictor v1、app reference anchors 與其檔案格式不變。App snapshot、多儀器 predictor 為後續工作。

沿用 `.importlinter` C14、C16，禁止 framework 反向載入 lab。
C20、C21 禁止 framework 依賴 migrations 或 scripts，升級能力由 caller 注入。
本次不新增 package 依賴規則。
通用 runner 與分析以公開接縫測試，lab 定義用直接 review、離線合成資料與 simulator 驗收。

## 後果

每台儀器可獨立校準，不必等待其他儀器的 integer 參考點。
Record 保存量測當時的 frame，component 改名或改接線不重寫歷史。
Operator 必須明確升級並重新核對校準，coil 的舊 half／int 不會重建為 offset。
格式與操作見 [歷史 v2](../storage-format-history/component-flux.md)、
[現行 v3](../storage-format-history/component-flux-v3.md)、
[record frame](../storage-format-history/flux-frame.md) 與 [升級指南](../flux-migration.md)。
