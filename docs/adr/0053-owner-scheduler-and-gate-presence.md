---
status: accepted
---

# 0053 — Owner scheduler seam 與 hardware gate presence

**狀態:** accepted（2026-07-12 contract freeze）。
**關聯:** [[0019]]、[[0021]]、[[0026]]、[[0044]]、[[0052]]。

## 背景

application core 對「Qt main thread」的殘餘依賴已收斂到少數機制點(`tests/gui/test_qt_import_boundary.py` 的 KNOWN_QT_DEBT 清單):core 真正需要的不是 Qt main thread,而是**「所有 State mutation 由單一 owner loop 序列執行」**這個不變式。具體殘餘:

- `gui/remote/rpc_endpoint.py` 的 `MainThreadDispatcher(QObject)`:IO thread → main thread 的 marshal 用 Qt queued Signal 實作。
- `gui/background.py`:`BackgroundExecutor` port([[0026]])的唯一實作是 Qt(QThread/QThreadPool),「完成後回 owner thread」靠 Qt 事件圈。
- 7 個 service 檔(session {connection,device} + app/main {run,save,analyze,post_analyze,staged_analyze})繼承 QObject 僅為了 completion Signal——它們的 async 執行早已走 `OperationRunner`。

另外,多前端 presence(「另一方正在跑 T1」)的資料基礎缺失:`RunBlocksHardwareGate` 的 `_ActiveLease` 只有互斥所需的 kind/owner_id/resource_id,不知道「被誰、為何、從何時」佔用。先例:Bluesky queueserver 的 lock 附 owner name + note,read-only API 永不受鎖。

偵察修正:先前計劃中的「persistence QTimer debounce 遷移」不存在——caretaker 只在 close 時 flush,core 內無 QTimer。scheduler 的初始消費者只有 rpc dispatch 一個。

## 決策

### OwnerScheduler port

```python
class OwnerScheduler(Protocol):
    def post(self, callback: Callable[[], None]) -> None:
        """Fire-and-forget:把 callback 排入 owner loop 執行。thread-safe。"""
    def call(self, callback: Callable[[], T]) -> T:
        """從非 owner thread 同步呼叫並取回結果(呼叫端阻塞)。thread-safe。
        在 owner thread 上呼叫是programming error,Fast Fail。"""
```

- port 放 `gui/session/ports.py`(與 `BackgroundExecutor` 並列)。
- **兩個 production/test adapter**:`QtOwnerScheduler`（移入 `gui/session/adapters/` 的 queued-Signal
  機制）與 `ManualOwnerScheduler`（thread-safe queue + owner-only pump）。inline direct scheduler只限
  same-thread unit test，禁止和ThreadPool組合，否則terminal callback會在worker寫State。asyncio adapter
  **不在本 ADR**——留待真實 Web/headless consumer。
- 真正消費者是 `remote/control_service.py`；它注入scheduler並保留
  `post + Event.wait(spec.timeout_seconds)` 的bounded RPC timeout。`call()`是adapter contract，不取代RPC
  timeout。`MainThreadDispatcher`隨之消失，`rpc_endpoint.py`清償Qt debt。
- **owner-loop 不變式改為機制守護**:Qt-free guard完整套用shared session與measure/autofluxdep/
  fluxdep/dispersive四個app State aggregates；autoflux direct semantic writes收進State mutators。

### BackgroundExecutor 第二實作與 background.py 重分類

- 新增 `ThreadPoolBackgroundExecutor`(純 threading;「完成後回 owner」透過注入的 `OwnerScheduler.post`)。
- `gui/background.py` 的 Qt 實作正式定位為 Qt runtime adapter:移入 `gui/session/adapters/`(或等價 ALLOWED 標記),從 KNOWN_QT_DEBT 清償。介面與呼叫端(`OperationRunner`)零改動。

### Service completion 通知去 Signal 化

7 個 service 檔的 Qt Signal 以 EventBus payload 取代。Run與DeviceSetup重用既有facts；只為
connection、device connect/disconnect、analyze/post failure、save resolved paths/errors補最小
in-process typed completion facts。error只帶string，不帶Exception/object，且completion facts不加入remote
serializer catalog。4處consumer改訂閱bus；QObject/Signal/parent全數移除，**不做event→signal轉接層**。

### Hardware gate presence

`ExclusionRequest`新增required nonblank `note`作為service→gate internal carrier；不改
`OperationRunner.__init__/begin`或`OperationSpec`。`_ActiveLease`與`register(...)`擴充三欄:

- `origin_kind`:發起者(取自 [[0052]] `EventOrigin.kind`:user/agent/system);
- `note`:發起 service 提供的人讀描述(如 `"run: T1 sweep (tab twotone/t1)"`);
- `since`:內部單調開始時間(顯示用途,不參與互斥判斷)。

變更時emit `GateChangedPayload(active=tuple[GatePresence,...])`；presence是list，因不同device lease可
同時存在。`gui_overview`透過read-only internal query投影`kind/origin_kind/note/active_for_seconds`，不曝露
raw monotonic epoch。Qt狀態列留待後續，read-only永不受鎖。

### Headless composition smoke test

新增subprocess測試:以 `ThreadPoolBackgroundExecutor` + `ManualOwnerScheduler` + offline mock組裝
session core（不建`QApplication`且阻擋Qt imports），pump owner queue跑success與cancel兩條完整operation
生命週期。這是core去Qt化的可執行定義，與import ratchet互補。

## 後果

- KNOWN_QT_DEBT 預期 13 → 2(剩 `app/main/app.py` composition root 與 `app/autofluxdep/controller.py`,各有明確後續歸屬)。
- Web/headless runtime 屆時只需新增 asyncio scheduler + transport,不再觸碰 core。
- gate presence 讓 Hybrid 模式(Qt + agent 並用)的互斥失敗從「被拒絕」變成「知道被誰、為何拒絕」。
- `cfg_binding.py` 與 `error_handler.py` 的 Qt 觸碰部分上移 ui 層(批次 3 工作單項目,機制不涉本 ADR)。

## 非目標

不做 asyncio adapter;不動 `OperationRunner` 介面;不做 Qt 狀態列 presence UI;不改 exclusion 判斷邏輯(誰擋誰的矩陣不變);不處理 autofluxdep controller 的 QThread/QTimer(Web Phase 4)。
