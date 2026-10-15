---
status: accepted
---

# ADR-0065 — Cfg 編輯模型與使用邊界

## 問題與決策

Qt、remote 與 Run 必須使用同一份 tab cfg。若 widget 持有唯一可編輯的樹，headless caller 就依賴 view 的存在。若 Run 重新解析 live source，caller 看到的值與實際執行值可能不同。

Measure 的 `TabFormResource` 擁有一份 Qt-free `FormState` 與固定來源。`TabFormResources` 管理建立、查找、操作禁令與撤銷。Qt、remote 與 Run 都使用同一資源。Frontend 只讀 schema、input、preview 與 errors，寫入走 resource 的公開入口。Form 決定 defaults、形狀、驗證與 core cfg 轉換，wire caller 不重建這些規則。

這條 Form 資源路徑適用於 measure tab。獨立 library、inspect 與 writeback editor 仍由 `CfgEditorService` 管理 `CfgDraft`，保留自己的 tree 與 editing codec。兩者的 read/write 契約不能互換。

## 共用機制與領域 owner

`gui.cfg` 擁有 Spec／Value、`CfgSchema`、input codec、binding、組裝、成品驗證及 lowering 機制。它不辨認 program discriminator、參數容器、ModuleLibrary 或 generation policy。`experiment.cfg_editing` 擁有 module／waveform 的 closed shape catalog、Spec factories 與 raw materialization policy。App 負責 runtime object normalization、選用子集合及 role defaults，composition 注入窄能力，converter 不寫 library。

`experiment.cfg_editing` 可依賴 Qt-free `gui.cfg`。Experiment package root 的公開 exports 按需載入，匯入 cfg_editing 不載入 experiment base、device 或 datafile。Library-entry conversion 與兩 app 的 normalization 尚未全部收斂，剩餘目標見 [cfg editing draft](draft/cfg-editing-boundaries.md)。

實驗 adapter 宣告 `Form(core_cfg, tree)`。Core annotations 定義可編輯形狀，tree 定義起始值、Slot、Template 與 lock。Open 與 Reset 計算 defaults。Memento 保存 Form input，不保存來源覆寫。Make_form、範本與固定來源的契約見 [[0073]]。Reset 可在同一資源內更換形狀與初值；成功後舊 editor state 不再承接寫入。

## Measure tab 的 input 與使用

Input 保存 ordinary JSON、顯式 `{expr: text}`、complex tags 與 model field names。普通字串保持 literal。Preview 提供 ordinary leaf values，不代表可以 Run。Form errors 包含帶路徑的診斷，合法但尚未完成的 input 保持可編輯。結構或來源錯誤拒絕整批，不保留成功前綴。

`tab.get_cfg`、`tab.edit_cfg` 與 `tab.reset_cfg` 回傳 schema、input、values、errors。Values 是 path/value 葉清單，errors 是 path/message 清單。Tab 以字串陣列指定 model-field leaf paths，range 分別寫 start、stop、n。Wire 不提供 CfgSchema kind、ref 子樹或 Valid/Invalid/Unavailable publication。

Read、preview、result 與 accept 不捕捉來源。通知期間允許讀取，禁止 mutation、accept、Run 及 close。Subscriber 故障記錄 traceback，不回滾已提交 input，也不阻止其他 subscriber 收到通知。

Run 只接受目前有效的 Form result，不收 cfg ref 或 expected revision，也不回退上次有效 cfg。Accepted config 深層隔離執行值與來源 provenance。後續 Edit、全域來源變更與 Reset 都不改已接受 Run。控制身份見 [[0070]]，hardware guard、lease 與 operation lifecycle 見 [[0066]]。

Tab 建立時就能 headless 編輯。Qt attach/detach 只管理 view 與訂閱。Idle close 撤銷舊 handle，重建同名 tab 建立新資源。Active Run 禁止同 tab 的人工 Edit、Reset、replacement 與 close，其他 tab 仍可編輯。Load 只載既有結果及分析來源，保留表單原輸入與固定來源。有效、Invalid 與未完成的 draft 都不被改寫，Load 不要求 current 有效，也不從 native cfg 推回 extras。Wire 回覆保留結果與分析準備 facts，不提供回填狀態。數據檔的 native cfg 從結果 Inspect 讀取。

## 固定來源與 Form library key

Open 與 Reset 捕捉一次 point、library、alias 表、cached device infos 與已發布 value sources。Edit、parameter_read、preview、result 與 Run 使用這份 capture，直到下一次 Reset。全域來源事件不刷新既有 tab。捕捉不輪詢硬體。

公開 `read_context(aliases=None)` 固定 point、library、storage 與全表。非空 aliases 只覆寫這次 capture 的具名列，保留 dependent authored links，不改全域表。省略、None 與 {} 使用全域列。EMPTY 且無覆寫時回 None，保留 context-free defaults，非空覆寫仍拒絕 no_context。

`tab.reset_cfg(aliases=...)` 是 tab 的單次覆寫入口，重新捕捉來源並恢復 defaults。覆寫不存入 alias 表或 memento。`tab.parameter_read` 只讀 resource.env.parameters，不收 aliases。`tab.run_start` 在任何 focus 或 operation 效果前拒絕非空 aliases，None 與 {} 使用畫面上的值與固定 capture。

RecipeSession.open_tab 的公開 aliases/reuse 用法不變。新 tab 與 reuse tab 都先 Reset，再設定參數與 Run。Target 對到 Reset 的 aliases，而非 preview 或 Run 的臨時解析。Recipe 的參數 schema 保持手寫，不從 Form 或 Python signatures 生成。

Form Slot 使用公開 ModuleRef/ModuleInline input。Key branch 保存 `{kind: ref, name: actual_library_key}`，inline branch 保存 `{kind: inline, module: Spec input}`。Nested key-or-inline 欄位用普通字串作 key。RecipeTab.use_library 寫實際 key，不建立舊 __ref editing intent。Locked Slot 只有 inline branch，use_library 在送 RPC 前拒絕選 key。

RecipeTab.read_field 回傳具名 RecipeFormField，包含 authored input、input_present、ordinary value、key/inline mode、actual key、Form lock、目前 child field names 與 directed errors。它不投影 CfgSchema kind，也不承諾 ref descendants 唯讀。Inline model 的路徑包含 module，例如 `modules.readout.module.freq`。Range count 的路徑是 n，set_sweep 的公開 expts 參數對到這個葉欄位。

實驗 cfg 依 [[0073]] 使用具名裝置欄位，例如 flux_dev、jpa_flux_dev，不再保存 dev role mapping。Form 的 AliasName 在 Open／Reset 解析 direct 名稱。使用 flux 的 cfg 必填 flux_unit，值為 A、V 或 native。Core 與 Notebook 不提供隱含 default，GUI 的初始值為 A。Recipe helper 只比 cached unit，成功後寫 caller 指定的具名裝置欄位與獨立 flux_unit，不換算、不連線或 setup。

Core 以 context.devices[cfg.flux_dev] 等具名欄位取 driver。缺名由實際使用者報錯；RunContext 不提供 setters。操作分工見 [[0067]]，Run 前 snapshot 與保存見 [[0063]]。Measure 不保留舊跨 tab source refresh／Unavailable 恢復路徑；使用時綁範本與缺來源的語義由 [[0073]] 擁有。

Form expression 的 `$alias` 與 `$alias.path` 沿固定 ParameterContext 解析，bare concrete paths 使用獨立 namespace。裸 `$alias` 是名稱字串，不參與數值算術或字串插值。缺列或未綁時 errors 指名 alias，不沿用舊解析值。Expression 不刷新來源。

## 獨立 editor 的 reference

獨立 editor 保留 CfgDraft 的 inline/ref mode、readonly descendants 與 common editing codec。Key selection 不複製 library 內容。Local reference conversion 由 cfg owner 處理，wire 不複製 shape、values 或解析規則。這些 tree 欄位不是 measure tab 的 Form DTO。

## Custom reference 的繼承

選擇 Custom 是明確的 resource-bound 命令。Cfg owner 從目前已發布內容準備完整新候選，best-effort 繼承相容輸入，再原子發布。切換 Gauss 至 DRAG 時，共同型別及單位的 length、sigma 可保留，新型別的固定值由 Spec 決定。不相容欄位使用新 shape 的初始值，nested linkage 保留其既定依賴。

這是操作便利性，不是相容補丁，也不保證候選符合量測條件。普通跨型別 edit 仍要求完整 payload，不因有繼承命令就隱含補值。Qt 不自行讀 owner input 或複製繼承規則。

## 獨立 draft 與尚未完成的範圍

`CfgDraft` 仍提供獨立 editor 的可編輯 field tree。Binding 列舉及解析 canonical dotted targets，remote 只投影這份 grammar。Measure tab 使用 Form JSON 與 string-array model-field paths。獨立 library/writeback batch 保留 fail-fast、non-atomic 成功前綴，不是 tab atomic edit 的另一入口。

Legacy `lower_finished_cfg` 接受 expression、reference shape 及 range 窄 ports。有 expression resolver 時會重新求值並檢查型別，可能在使用時讀來源；這不是 measure tab 的 acceptance 路徑。Tab Run 使用已解析並接受的 values，不在 legacy lowering 時偷換成 live source。

Writeback 以 opaque draft 固定 ParameterContext 與分析來源的 run_id。Selected Apply 先準備全部 proposals，再追加 accepted 並依序提交 target setters，首錯保留成功前綴，不提供跨檔原子性或 crash durability。Autofluxdep 的 placement refresh、resource lifetime、固定 Run acceptance，以及 library／writeback Apply 的剩餘收斂仍見 [cfg editing draft](draft/cfg-editing-boundaries.md)。本篇不宣稱整個 multi-app Controller 工作完成。

## 取捨與相鄰責任

單一 tab resource 讓 GUI、remote 與 Run 共用 Form input、驗證與固定來源，代價是 caller 想使用最新參數時必須明確 Reset。這比 Run 偷換 live source 更容易核對畫面與執行值。Source basis 描述 provenance，不是 cfg admission 門檻。保留獨立 draft，避免把 library Apply 誤當成 tab edit。

參數容器／library 寫入權由 [[0067]] 定義；保存與 crash durability 歸 [[0063]]；Autofluxdep run-start base 與逐點 patches 歸 [[0062]]；wire、guard、delivery failure 見 [[0068]]。模組入口見 [cfg owner](../../lib/zcu_tools/gui/cfg/README.md)、[experiment editing owner](../../lib/zcu_tools/experiment/cfg_editing/README.md)、[measure app](../../lib/zcu_tools/gui/app/measure/README.md) 與 [Autofluxdep app](../../lib/zcu_tools/gui/app/autofluxdep/README.md)。

舊篇 [[0008]] 至 [[0012]]、[[0037]]、[[0045]]、[[0046]]、[[0050]] 及 [[0051]] 保留尚適用的局部契約。它們的 tab draft、雙樹、live Run 或成功前綴敘述不覆蓋本篇的 measure tab resource 契約。
