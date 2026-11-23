---
status: accepted
---

# 0002 — 並發感知：資源版本表與 off-main blocking handler（handle 模型沿革）

本篇記錄資源版本表與 RPC off-main handler。遠端寫入的控制身份見 [[0070]]，operation handle 的現行規則見 [[0066]]。
**關聯：** Permit／Lease 見 [[0001]]；handle、lease、取消與關閉的現行分界見 [[0066]]；shared transport 見 [[0068]]，external-refresh Reaction 見 [[0067]]。

## 脈絡

Qt View 和 remote RPC agent 都會操作同一份 GUI 狀態。資源版本記錄狀態變動。等待 operation 完成的 RPC handler 必須讓 owner loop 繼續處理完成通知，否則等待會卡住。

## 決策

資源版本與 RPC 等待是兩項獨立的契約。

### 1. 資源版本表（只服務 guard）

**規則。** 節名中的「只服務 guard」是舊用途。現行 `VersionTable` 不作 RPC 寫入門檻。GUI 的 `State` 持有版本表。每個資源各有單調遞增的整數計數器，不用牆鐘。資源 owner service 在 State owner loop 寫入資源並推進版本。同步操作由 mutator 寫入，背景操作由回到 owner 的 terminal policy 寫入。Worker 不直接提交 State。讀取後同步快取時，只有值實際改變才推進版本並發出變更通知。

**例子。** 背景讀取裝置資料後，`DeviceService` 回 owner 比對 `device:<name>` 的快取。例如讀回的值沒變，就不推進該版本，也不發通知。見 [session owner](../../lib/zcu_tools/gui/session/README.md#結構)。`context`、`soc` 也各有計數器。Tab 資源用 `tab_id` 分開計數，例如 `tab:<id>` 記存在版本，`tab:<id>:cfg` 和 `tab:<id>:result` 分別記 cfg 和結果版本。

**邊界。** 版本在資源寫入處推進，不靠 event origin、emit 或 lease release 的順序。各 app 擁有資源 key；計數器退役及同名重建見 [GUI owner README](../../lib/zcu_tools/gui/README.md#resource-version-lifecycle)。Measure RPC 不保存每條連線的 seen map，也不以讀取建立寫入基準。控制身份及現行 admission 見 [[0070]] 與 [ADR-0068 的一致性契約](0068-remote-transport.md#method-與一致性)。

**依據。** Git `3c2ce256e` 記錄資源版本表取代舊 origin guard，`b85f54f6a` 記錄控制身份接替 per-connection seen guard。版本表初版的原始討論未入庫。

### 2. Operation handle（歷史設計）

已由 [[0066]] 取代，見[〈問題與決策〉](0066-operation-lifecycle.md#問題與決策)；[原文](retired/operation-handle.md#2-operation-handle歷史設計)僅供追溯。

### 3. off-main blocking handler

**規則。** `MethodSpec.off_main_thread` 預設為 `False`。設為 `True` 時，dispatcher 在 IO worker thread 執行 handler，不送到 State owner loop。

**例子。** `operation.await` 在 IO worker 對 thread-safe channel 做有期限的等待。Owner loop 繼續處理 operation 完成通知。

**邊界。** Off-main handler 只做 thread-safe 等待，不得讀寫 owner-thread state，包括版本表。一般 RPC handler 仍在 State owner thread 執行。見 [measure remote owner](../../lib/zcu_tools/gui/app/measure/remote/README.md#authority-and-dispatch)。

**依據。** Phase 93 把等待移出 owner loop，因為舊 `device.wait_setup` 曾阻塞 Qt event loop，使完成 signal 無法送達。原始討論未入庫；死鎖修正見 Git `1e91a3538`，現行 await 分工見 [[0066]]。

## 三層分工（脊椎）

- **GUI** 持有版本表，在 owner dispatch 檢查目前 holder。資源版本可供來源、結果替換與 provenance 使用，不用來授權 RPC 寫入。
- **MCP** 轉送單次 RPC，維護 catalog 與 operation handle 的對應。不為解鎖寫入而隱藏預讀，也不在斷線後自動重送 mutation。見 [measure MCP owner](../../lib/zcu_tools/mcp/measure/README.md#連線與觀察)。
- **Agent** 讀取快照與 operation 狀態。讀取是為了核對現況，不是為了建立寫入許可。Poll／wait 的呈現見 [[0068]]，operation 生命週期見 [[0066]]。

## 演化（被取代的設計，保留脈絡）

Phase 92/93 的 origin tracking、Phase 120c 的 agent 事件工具移除，以及後來的 seen guard，分別可從 Git `1e91a3538`、`a42cb447b`、`572023951` 追溯。現行 EventBus、診斷與操作觀察的分工見 [ADR-0068](0068-remote-transport.md#事件與錯誤)。

## 替代方案與否決理由

舊方案的比較見 Git `dc74c1e08` 的本文。現行版本更新跟隨資源寫入，不跟隨事件通知；等待需要 owner callback 才能完成的工作時，不阻塞 owner loop。

## 範圍

本篇只定義資源版本與 off-main handler 的局部契約。Operation 執行範圍與 owner 分界見 [[0066]]。
