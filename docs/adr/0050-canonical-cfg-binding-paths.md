---
status: accepted
---

# 0050 — Canonical cfg binding paths

## 脈絡

Cfg mutation 和 listing 若各自判斷 field subtype，scalar types、reference keys、sweep edges 和 suggestions 就會分歧。`gui.cfg.binding` 擁有獨立 draft 的 target grammar。Measure tab 使用 cfg resource 的 string-array paths 與 atomic edit，不使用 draft batch 的成功前綴契約，見 [[0065]]。

## 決策

Binding 以 nominal `SettableTarget` 共用 listing、resolve、setting、validation、suggestions 與 migration hints。每個列出的 path 都能解析成相同 kind、value type、current value、choices 和 `affects_path_shape`。未列出的 alias 不執行。

Scalar 使用 dotted leaf。Sweep 直接以 `start|stop|expts|step` 結尾，centered sweep 使用 `center|span|expts|step`。Reference key 使用 `.ref`，children 直接下鑽。舊 `.sweep.*` 和 `.value.*` 拼法在 mutation 前拒絕，並提供已核對的 replacement。Target tree 建立時拒絕無法表示或保留的 field key。

Qt sweep widgets 使用 leaf targets。Agent 可在 sweep parent 使用 nominal aggregate target，交付完整 sweep object。兩者共用 SweepEditor／CenteredSweepEditor 的驗證與正規化，remote 不複製 subtype grammar。

`gui.cfg` 擁有 custom-reference-key make／parse／query helpers。Binding 正規化可接受的 bare labels，內部 tag 只用於 persistence，不依賴 session policy。`ValueRef` 在 app seam 解析且只用於 scalar target。

Binding path 指向 cfg 欄位，不是參數容器路徑。欄位中的 `$alias` 名稱、`$alias.path` expression 或 module slot 由固定 ParameterContext 解析；library key 使用具體元件路徑。Alias table 與具體參數路徑的權責見 [[0037]]／[[0067]]，不在 binding grammar 推論。

## 完整觀察與 batch

完整 cfg 讀取使用 `CfgDraft.observe()`。`CfgNodeObservation` 包含 locked literals、cached options、active reference children、validity，以及 raw／resolved／error value carriers。Binding 回傳 detached data，不重新解析來源。Settable listing 和 persistence encoding 都不能代替完整 observation。

Remote 將 observation 序列化並套 prefix read policy，不 import field/editor classes 重建規則。只有無 prefix 的成功完整讀取揭露整個 cfg revision。Unknown objects 和 nonfinite values 使序列化失敗，不轉成 display string。Prefix 或裸版本不證明完整讀取。

獨立 draft batch 依序執行，首錯停止，不具原子性。Reference switch 可以先暴露下一個 path。Shape edit 才計算 path sets，在第一筆 shape edit 前和成功後各計算一次。回覆是排序後的最終 net diff，A→B→A 為空。失敗保留成功前綴，傳遞原 typed error，不做 after diff。每筆成功 edit 的版本更新與 [[0068]] 的 subscriber 投影保持原契約。

## 後果

Listing 和 resolution 共用一份 grammar。Agent 直接複製 listing paths，舊拼法得到可定位的提示。獨立 draft 的成功前綴不代表 tab resource 的 atomic batch，也不代表 Apply 的跨檔原子性。

本篇延續 [[0008]]、[[0045]]、[[0046]] 的局部契約；跨 owner 分工見 [[0065]]，remote 邊界見 [[0068]]。
