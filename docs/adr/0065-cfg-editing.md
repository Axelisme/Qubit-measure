---
status: accepted
---

# ADR-0065 — Cfg 編輯模型與使用邊界

## 問題與決策

量測 GUI、agent 和 Autofluxdep 都要編輯 Spec／Value 設定。若 widget 擁有唯一可編輯的樹，沒有開啟 widget 的 agent 或 writeback 就不能共用同一份設定。若通用 cfg core 辨認 program 形狀或自行取得整個 app context，新增實驗會改動不擁有該領域政策的層。

`gui.cfg` 擁有 Qt-free 的 Spec／Value、`CfgSchema`、binding、codec、組裝與 finished-cfg validation／lowering 機制；`gui.widgets.cfg` 只呈現 caller 提供的 `CfgDraft`，detach 不關閉 draft。資源的 app owner 管理其編輯 lifetime，並提供 expression、reference 與 option 來源。measure 的 `CfgEditorService` 持有 tab、library entry 及 writeback 項目的 headless draft；widget 與 agent 操作同一個 draft。writeback 以 opaque draft 封裝各項 session identity，提交仍交給 `ContextService`。Autofluxdep 的 placement value tree 由 workflow owner 保存，Default cfg 與 Generation overrides 保持不同用途；目前 form 建立兩個局部 draft，編輯時由 Controller 接收合併後的 value tree。它沒有共用 measure 的 editor service。這個 form-local lifetime 與已核准的跨 frontend resource-owned 目標不同，見 [cfg draft](draft/cfg-editing-boundaries.md)。

通用 core 不解讀 program discriminator、MetaDict、ModuleLibrary 或 generation policy。`experiment.cfg_editing` 擁有 module／waveform 的 closed shape catalog、Spec factory 與 raw materialization policy；app 仍負責 runtime object normalization、選用的子集合與 role defaults。composition 將窄能力接到 consumer，converter 不寫 library。其 import 會載入 `experiment` package 的 base 依賴，並非完全不載入 experiment。實驗 adapter 的 context-free definition 在建立 fresh cfg 時才解析 deferred defaults；restore 與 refresh 不重跑 resolve-once seed。Autofluxdep 的 logical paths、generation plan 和 run-time patches 仍由其 workflow owner 管理（[[0062]]）。

`gui.cfg.binding` 擁有可列舉且可解析的 canonical target path；remote 只將它投影為 wire 形狀，不建立另一套欄位 grammar。共用 lowering 接受 expression、reference shape、range 三個窄 port，app 提供實際來源與 domain policy。編輯中的 schema 可不完整；成品使用者在 lowering 前檢查結構與值。當前 lowering 對已解析的 EvalValue 使用 embedded result，沒有 result 才從傳入的 expression resolver 求值；linked reference 的 shape 會查詢 resolver，內容仍取 embedded value。這不是「Run 已接受指定 editor revision」的保證。measure 的 edit batch 目前逐筆修改 live draft，錯誤保留已成功的前綴；不能把 net path diff 或一次 Context write call 當成原子編輯或原子 Apply 的證據。

session source owner 以唯讀 lookup 提供小量跨來源值，只有 composition 與來源 owner 註冊 provider；device lookup 讀 cached observation，不輪詢硬體。來源 owner 通知外部變更，measure editor service 對其 draft 觸發 expression／reference／option refresh；widget 不擁有其生命週期。Autofluxdep 目前由開啟的 node form 對其局部 draft 處理刷新事件，未開啟 form 時沒有等價的 placement-owned refresh。Linked ref、EvalValue 和 resolve-once 的現行差別是：linked ref 保留 key 與內嵌內容，可在 refresh 時更新投影；EvalValue 保存 expression 及解析結果；`ValueRef` 或 fresh seed 在輸入時讀取一次，保存普通 direct 值，不跟隨來源。現行 overridden reference 在部分 missing-key 路徑會轉成 custom key，且 lowering 仍查原 linked key；不能將核准的「override 解除此層來源依賴」當作現況。refresh／failure／relink 及指定 revision 使用的目標契約見 [cfg draft](draft/cfg-editing-boundaries.md)。

## 取捨與相鄰責任

單一通用資料機制減少兩個 app 對 path、codec 和 lowering 的重複實作，代價是每個 app 必須提供窄 port 並管理自己的資源。通用 renderer 不取得 runtime policy，也不強制 Autofluxdep 使用 measure service。當前 app 的編輯與使用時機尚未完全一致；此篇不把 UI auto-commit、使用時 live 解析或 batch 成功前綴提升為未來跨 app 的規則。

md／ml 寫入權由 [[0006]] 定義；保存責任與 crash durability 歸 [[0063]]；workflow 的 run-start base 與逐點 patch 歸 [[0062]]。局部資料形狀、binding 操作與 codec 見 [GUI cfg owner](../../lib/zcu_tools/gui/cfg/README.md)、[experiment editing owner](../../lib/zcu_tools/experiment/cfg_editing/README.md)、[measure app](../../lib/zcu_tools/gui/app/measure/README.md) 與 [Autofluxdep app](../../lib/zcu_tools/gui/app/autofluxdep/README.md)。舊篇 [[0008]]–[[0012]]、[[0037]]、[[0045]]／[[0046]]、[[0050]]／[[0051]] 保留未完全遷移的局部契約；與本篇或 draft 衝突的舊行為不構成新的共用保證。
