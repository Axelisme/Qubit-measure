---
status: accepted
---

# ADR-0065 — Cfg 編輯模型與使用邊界

## 問題與決策

Qt、remote 與 Run 必須使用同一份 tab cfg。若 widget 持有唯一可編輯的樹，headless caller 就依賴 view 的存在。若 Run 重新解析 live source，caller 看到的值與實際執行值可能不同。

`gui.cfg` 提供 Qt-free 的資料機制。`CfgResource` 擁有 input tree、identity、revision、解析及 publication。Measure 的 `TabCfgResources` 擁有 tab 資源的建立、查找、操作禁令與撤銷。Qt 取得 editing handle，Run owner 取得 acceptance 能力，remote 將請求送到同一資源。Controller 組裝這些能力，不逐項轉送 cfg edit commands。

這條資源路徑適用於 measure tab。獨立 library、inspect 與 writeback editor 仍由 `CfgEditorService` 管理 headless draft。Autofluxdep 的 Default cfg 與 Generation overrides 仍由 node form 建立局部 draft，workflow owner 保存 placement value tree。這些用途沒有因 measure tab 切換就全面遷移。

## 共用機制與領域 owner

`gui.cfg` 擁有 Spec／Value、`CfgSchema`、input codec、binding、組裝、成品驗證及 lowering 機制。它不辨認 program discriminator、MetaDict、ModuleLibrary 或 generation policy。`experiment.cfg_editing` 擁有 module／waveform 的 closed shape catalog、Spec factories 與 raw materialization policy。App 負責 runtime object normalization、選用子集合及 role defaults，composition 注入窄能力，converter 不寫 library。

`experiment.cfg_editing` 可依賴 Qt-free `gui.cfg`。Experiment package root 的公開 exports 按需載入，匯入 cfg_editing 不載入 experiment base、device 或 datafile。Library-entry conversion 與兩 app 的 normalization 尚未全部收斂，剩餘目標見 [cfg editing draft](draft/cfg-editing-boundaries.md)。

實驗 adapter 的 context-free definition 只在 fresh cfg 或明確 reset 時解析 defaults。Restore 保存輸入，source refresh 不重跑 seed。Definition 在資源 lifetime 中固定；需要另一份 definition 時由 app 建立新資源並撤銷舊 identity。

## Measure tab 的 publication 與使用

Input 保存 direct value、raw text、expression、reference 及 range 的編輯意圖。Input 中的解析結果不作為可信結果。資源先準備隔離的完整候選，再同步解析並發布一個 revision。成功的同值命令與空 batch 也增加 revision。被拒絕的 batch 不保留成功前綴。

Publication 同時包含 `CfgRef`、Valid／Invalid／Unavailable、tree、source basis 及 diagnostics。合法未完成輸入可以成功發布 Invalid。必要來源未就緒或 source refresh 故障發布 Unavailable，不保留舊 Valid 冒充新結果。Edit／Reset 的非預期準備故障則保留原 publication。

Observe、watch 及 accept 不刷新來源。Watch 先註冊再交付初始 observation；unsubscribe 停止後續通知。通知期間禁止 mutation、accept、Run 及 close。Subscriber 故障送診斷，不回滾已提交 publication，也不阻止其他 subscriber 收到結果。

Run caller 明示觀察到的 `CfgRef`。只有同 identity、同 revision 的 Valid publication 可以接受。Stale 或非 Valid 不提交 operation，不刷新、不換版、不重試。Accepted config 深層隔離 values 與 source basis，worker 和 artifact 使用固定資料，後續 source 更新不改執行值。Cfg acceptance 不替代 hardware guard、lease 或 operation lifecycle，見 [[0066]]。

Tab 建立時資源已可 headless 編輯。Qt attach／detach 只管理 view 與 watch，不建立或撤銷 cfg。Load backfill 在同一資源準備完整 Valid 候選，成功保留 identity 並增加 revision，失敗保留原 input 與 publication。Idle close 撤銷舊 handle，重建同名 tab 使用新 identity。Active Run 禁止同 tab 的人工 edit、reset、replacement 及 close，其他 tab 與自動 source publication 不受此禁令阻擋。

## 固定來源與 reference

Source owner 發布本地來源及版本。解析使用固定 `CfgResolution`，不輪詢硬體或做外部 I/O。Device lookup 使用已發布 cache。Source 更新先準備並安裝所有受影響 cfg 的 publication，再通知；某個 cfg 解析故障不撤銷其他資源或 source 的更新。

Linked reference 隨固定來源更新這層內容。Override 解除這層對原 key 的依賴，保留自身 shape 與 input，nested linked reference、expression 及 asset 依賴仍各自存在。明確 relink 才重新建立來源依賴。Expression 保留文字與動態依賴，refresh 更新解析結果；resolve-once direct value 不重讀來源。

`gui.session.expression` 共用 simpleeval 的受限數學引擎。Cfg expression 中 `$` 標記的引用在當次寫入以已發布來源捕捉，再以 typed literal 取代該位置，未標記引用保持動態。捕捉失敗整批拒絕，不保存待未來解析的 `$`。它仍保存 expression，不新增 capture 持久型別。精確語法與錯誤邊界見 [resource contract](draft/cfg-resource-contract.md)。

## Custom reference 的繼承

選擇 Custom 是明確的 resource-bound 命令。Cfg owner 從指定 revision 的已發布內容準備完整新候選，best-effort 繼承相容輸入，再原子發布。切換 Gauss 至 DRAG 時，共同型別及單位的 length、sigma 可保留，新型別的固定值由 Spec 決定。不相容欄位使用新 shape 的初始值，nested linkage 保留其既定依賴。

這是操作便利性，不是相容補丁，也不保證候選符合量測條件。普通跨型別 edit 仍要求完整 payload，不因有繼承命令就隱含補值。Qt 不自行讀 owner input 或複製繼承規則。

## 獨立 draft 與尚未完成的範圍

`CfgDraft` 仍提供獨立 editor 的可編輯 field tree。Binding 列舉及解析 canonical dotted targets，remote 只投影這份 grammar。Measure tab 則使用 string-array paths 與共同 editing codec，兩種 read/write 契約不能互換。獨立 library／writeback batch 保留 fail-fast、non-atomic 成功前綴，不是 tab atomic edit 的另一入口。

Legacy `lower_finished_cfg` 接受 expression、reference shape 及 range 窄 ports。有 expression resolver 時會重新求值並檢查型別，可能在使用時讀來源；這不是 measure tab 的 acceptance 路徑。Tab Run 使用已解析並接受的 values，不以 legacy lowering 的 live read 保證指定 revision。

Writeback 以 opaque draft 封裝 session identity，提交交給 context write owner。一次 `ContextWritePort` 呼叫不能證明 selected Apply 的全部失敗原子性或 crash durability。Autofluxdep 的 placement refresh、resource lifetime、Run 接受版本，以及 library／writeback Apply 的剩餘收斂仍見 [cfg editing draft](draft/cfg-editing-boundaries.md)。本篇不宣稱整個 multi-app Controller 工作完成。

## 取捨與相鄰責任

單一 tab resource 讓 GUI、remote 與 Run 共用版本、驗證及 publication，代價是 app 必須提供固定來源、管理 lifetime，caller 必須明示觀察版本。保留獨立 draft 用途避免將 library Apply 或 workflow patches 誤當成 tab edit。

md／ml 寫入權由 [[0067]] 定義；保存與 crash durability 歸 [[0063]]；Autofluxdep run-start base 與逐點 patches 歸 [[0062]]；wire、guard、delivery failure 見 [[0068]]。模組入口見 [cfg owner](../../lib/zcu_tools/gui/cfg/README.md)、[experiment editing owner](../../lib/zcu_tools/experiment/cfg_editing/README.md)、[measure app](../../lib/zcu_tools/gui/app/measure/README.md) 與 [Autofluxdep app](../../lib/zcu_tools/gui/app/autofluxdep/README.md)。

舊篇 [[0008]] 至 [[0012]]、[[0037]]、[[0045]]、[[0046]]、[[0050]] 及 [[0051]] 保留尚適用的局部契約。它們的 tab draft、雙樹、live Run 或成功前綴敘述不覆蓋本篇的 measure tab resource 契約。
