# 0082 — Component flux frame 與明確升級

日期：2026-10-27
狀態：Accepted；下列標示 advisor 的實作細節為 Review: pending。
相關：[[0063]]、[[0069]]、[[0072]]、[[0081]]

## 脈絡

一顆 qubit 可能同時受到 local 線、global coil 與其他 DC 線影響。
分開保存 local 校準與 mux coil 校準，無法表示一組同時的參考點。
只保存儀器值的 record，也無法在 flux jump 或重新校準後還原當時的 flux。

Component schema 與 YAML 容器各自演進。
讀取時猜格式會把相容邏輯分散到每個 caller，因此舊文件經明確命令升級。

## 決策

### Component 以 integer 為參考點

每顆 qubit 保存 `flux.<device>.{half, integer, period}`，預設為空 mapping。
每個 device 使用自己的原生儀器單位，單位取自 device info。
三欄皆為有限實數，period 非零。負 period 表示增加 bias 會減少 flux。

所有 contributing devices 同時停在各自 integer 時，qubit 的 reduced flux 為零：

```text
phi = sum((value[d] - integer[d]) / period[d])
```

每個 half 與 integer 的距離須為 `abs(period)/2`。
載入檢查數值形狀，使用處檢查三欄 spacing 並指出 device。
Qubit 保留 `flux_source` 作表單預設掃描來源，不再保存舊四個 scalar flux 欄位。
Mux group 保留自己的 `flux_source`，各 qubit 自己保存對該 device 的校準，不再有 `coil_flux`。
`coil` 不是保留別名，mapping key 一律是實際 device 名稱。

使用者於 2026-10-27 指定：「ref_flux就不用了，固定就好。我反而考慮flux.<device>.{half/integer/period}這種格式。」
同日修正參考點：「我認為你ref_flux使用integer比較好。」
對 device 名稱限制指出：「那device那邊可能要做對應的檢查，就是給device名加上限制」。

名稱與註冊共用 `cfg_model.validate_device_name`，拒絕 ASCII 點號、空字串與全空白。
其餘名稱原樣保留。拒絕空白名稱是 advisor 裁定，Review: pending。
Device membership 按使用處的資料檢查，不新增 setup catalog。
Form 使用 detached device 選項，writeback 使用 Run snapshot，分析使用該列 devices。
這些 membership 與初值接線細節為 advisor 裁定，Review: pending。

### Fluxdep 分開保存原始擬合與參考校準

掃描 device d 時，其他 devices 固定在 Run snapshot 的值。
原始擬合 integer 為 h，寫回時換算：

```text
integer[d] = h + signed_new_period[d] * sum_other((value[o] - integer[o]) / period[o])
```

Half 作同樣平移。Point 只更新被掃 device 的完整 entry。
Period 只更新大小，保留原方向，首次校準使用正方向。
Form 與 picker 初線作反向平移，得到其他 devices 當下值對應的實際 half／integer。

同次接受另 append ordinary record，保存原始 half、integer、正的 period 與被掃 device。
Record 的 devices 來自 Run 開始時的完整 snapshot。
Flux frame 來自 Analyze 時該 Run 的固定 parameter capture，不混入本次新校準。
若另一個 contributing device 在同一次 Run 也被掃，拒絕該 qubit 寫回。
這些 snapshot、同掃拒絕與初次正方向細節為 advisor 裁定，Review: pending。
接受時不重讀目前 point、alias 或 live device。

### Record 與分析

Lab Record Policy 要求 T1、T2、frequency、fluxdep 與 workflow records 保存 `flux_frame`。
Notebook 作者在量測前捕捉 frame，經既有 `SavedRun.accept(records=...)` 明傳。
Lab helper `record_flux_frame` 將捕捉的 qubit flux 區塊與同次 Run snapshot 配對，作者也可組裝自己的定義。
Snapshot 存在時，非空 frame 的每個 device 都須有有限、非 bool 的 numeric value。
缺值或無效時，lab writer 在組 proposal 時以 `ValueError` 指名 device，不查 setup 或 live 值。
空 frame 合法。整個 snapshot 缺席時，一般 GUI adapter 省略 record，workflow 保留 `devices=null`。
Fluxdep writeback 仍因無法換算而報錯。原始 Save 不受此檢查限制。
Notebook 作者在 Run 後組 records 前做相同核對，缺 snapshot 不組 record。
這項既有 use-site 契約的補齊為 advisor 裁定，Review: pending。
Framework append owner 加 `component` 信封，不定義 flux 物理意義。

`flux_from_frame` 逐列依 explicit flux、新 row-frame、caller fallback 的順序換算。
Nested frame 的 keys 決定該列 contributors。Dotted CSV 以該列每個 device 任一非 null frame 欄位決定 membership。
全 null 表示不存在。攤平無法區分不存在與原 frame 有 key 但全部為 null。
缺用到的 device 值、integer 或 period 時報錯，不跳過該線，也不退回 fallback。

歷史改名的 records 可混合分析，各列使用當時名稱。
改接線後刪除 component 的對應 flux entry 並重做 fluxdep，歷史 records 不變。
只有舊 flat frame 且沒有 explicit flux 的列，由作者確認單位後明傳 caller fallback。
獨立 `resolve_record_flux`、`validate_record_table_v2` 與單 device `RecordFluxFrame` 保留既有 flat 座標契約。
這些分析 API 的保留與 membership 細節為 advisor 裁定，Review: pending。

### Component 文件版本與升級

Setup、point 與 entry template 共用整份文件的 `schema_version`。
Lab runtime 現在為 2，與移除舊 component 欄位及更新模擬範本同 commit 交付。
Runtime 只接受目前版本。缺版或不同版本時，錯誤列出文件版本、目前版本與長期 upgrade 命令。
讀取不升級，也不改檔。

使用者於 2026-10-27 說：「每次改component都會讓舊資料不支援很頭痛」。
對整份文件版本與通用 runner／lab steps 的建議回覆：「12採用建議」。

`resources.parameter_upgrade` 擁有版本鏈、dry-run、備份與寫入機制。
Lab 擁有目前版本、完整分類與純函式步驟。
未標版文件只有在明確 upgrade 且完整分類成功後才視為 v1。
Partial／corrupt 文件不寫入，其餘文件繼續。Apply 先備份原 bytes，current 文件驗證後跳過。
優先新增有預設值的欄位，結構變更才新增一步。
歷史步驟只在使用者確認資料都已升版後刪除。

Flux v1→v2 共用一份 lab 純轉換。它把 local 與 mux 校準合併到 qubit flux，保留原數字，不推測缺少的 source。
舊校準的同時參考點是假設值，operator 須重新校準。
Shipped storage mapping 在完整 seed／value batch 組成 EditView 前轉換，直接驗證並發布 v2。
不先發布 v1 再升級，也不改 detached staging。
Storage-only lab callback 將完整有效、明示來源的歷史校準一次性取代同 device 的繼承校準，其他 entries 保留。
它先確認整個 incoming batch，local／mux 同批衝突、部分校準與缺來源仍拒絕，不能逐次刪 entry 變成最後寫入者勝出。
Native 升級仍拒絕混合文件。這個 storage-only 覆寫語意為 advisor 裁定，Review: pending。
長期入口為 `scripts/upgrade_parameters.py`，不提供獨立 `migrate_flux`。
`migrations/` 仍是可刪除的舊 repo 匯入工具，長期升級鏈不依賴它。
同一 CLI 的 record 範圍只按 operator 宣告轉換無名 device 值，不補歷史 frame 或 component。

## 邊界與可執行契約

沿用 [[0072]] 的 lab→framework 方向。Flux 數學只收純 analysis 型別，不讀 component。
Lab helper 負責 component 到數學型別的轉換，Measure proposal 組裝留在 lab measure support。
獨立 predictor 模型與 fluxdep app reference anchors 的同名欄位保留原語義、格式與版本。
Runtime PredictorService、predictor 檔與 fluxdep app anchors 的多 device 輸入不在這個改版範圍。
Workflow FrequencyModelForm 的初值仍需移除 component 舊欄位讀取。它按 flux_source 選新 entry，
以 Form capture 的其他 device 值換算 actual half，保留該 entry 的 signed period。
缺來源或校準時保持必填空白，frame 錯誤保留 device 名稱。EJ／EC／EL 的初值不變。
FrequencyModel 的欄位與單 device affine 行為不變。這個初值範圍釐清為 advisor 裁定，Review: pending。

沿用 `.importlinter` 的 C14、C16，禁止 framework 反向載入 lab。
C20、C21 禁止 framework 依賴 migrations 或 scripts，升級能力由 caller 注入。
未新增 package 依賴規則。通用 runner 與分析以公開接縫測試，lab 定義用直接 review、離線合成資料與 simulator 驗收。

## 後果

Record 自帶量測當時的 frame，不需要以目前校準猜舊 flux。
Operator 必須明確升級舊文件，並檢查多 device 參考點假設。
Component 改名或改接線不重寫歷史。
格式實例、辨識與修復見 [component flux 歷程](../storage-format-history/component-flux.md)、
[record frame 歷程](../storage-format-history/flux-frame.md) 與 [升級指南](../flux-migration.md)。
