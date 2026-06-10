---
status: accepted
---

# Device 狀態下放 State（SSOT）+ DeviceService 純 driver/worker + persistence 為 State 投影

**狀態：** accepted。實作於 Phase 96–98（gui2）。

## 脈絡

`DeviceService` 私藏了一份 live device 狀態 `self._snapshots: dict[str, DeviceSnapshot]`，在主線
terminal slot 直接寫它。這違反既有核心不變式（**所有 State 寫入只在 Qt 主線；worker 唯一允許的副作用是
emit Qt signal**，見 ADR-0002）：那份本該屬於 `State`、被 version table 治理的狀態，被
service 繞過 State 持有了。

連鎖症狀：

- `remember`（這個 device 連上後是否要記住）只活在 connect 的 `req.remember` 瞬時屬性，**沒進 snapshot**。
  於是 startup persistence 無法從狀態 derive「該記住哪些 device」，被迫由 connect/disconnect 事件把 `req`
  一路餵給 Controller → StartupService 做增量 `add_device`/`remove_device`。
- `remember` 的判斷散在兩處（DeviceService `_on_disconnect_succeeded` 內 + Controller handler 內）= 漂移風險。
- persistence 協調邏輯散在 Controller 的 `_on_device_connected`/`_on_device_disconnected`/`forget_device`
  handler，與「Controller 是薄 façade」的契約矛盾。

> 註：ADR-0002 曾述「device 在 gui 外的 `GlobalDeviceManager`，本就不經 State」。本 ADR **修正**該前提：
> device **狀態** 進 State（bump 移入 State mutator）；`GlobalDeviceManager` 只持 **live driver 物件**。

## 決策

**State 是 device 狀態的 SSOT；DeviceService 退化為 driver 持有者 + worker 協調者；persistence 是 State 的投影。**

### 1. State 持 device 狀態

- 新 frozen `DeviceState`（`state.py`）：name/type_name/address/status/**remember**/info/error。`remember`
  從瞬時 request 屬性**升格為持久狀態**。`info` 是 `BaseDeviceInfo`（可序列化 value，非 driver），故進 State。
  **無 `progress` 欄位**：setup progress 是 live telemetry，由 `ProgressService`（以 operation token 為 key）持有，經 `operation.progress`（by operation_id）查詢。
- `State.devices` + mutator（對齊既有 tab mutator）：`put_device` / `set_device_status` /
  `set_device_info` / `set_device_remember` / `remove_device`（`drop_prefix`）。每個**語義寫入**內
  `version.bump("device:<name>")`。
- **`refresh_device_info_cache` 條件式 bump**：`get_device_info` 在讀取時把 driver 回來的最新 info 快取回
  State。**值與快取相同 → 純同步，不 bump、不 emit**（否則一次純讀會 spurious 推進版本號、誤使其他 client
  的 `expected_versions` 失效）；**值不同（pydantic `!=`）→ driver 值在外部變了 = 真實狀態變化，bump
  `device:<name>` 並回 True 讓 caller emit `DEVICE_CHANGED`**。原則精確化：bump = 狀態真的變了；不 bump 的是
  「值未變的快取同步」，不是「所有讀取路徑」。

### 2. DeviceService = driver/worker

- 移除 `self._snapshots`；所有讀寫走 State mutator/query。
- `DeviceSnapshot` 變 **read-time projection**（`_project(DeviceState)`），保留原 dataclass 形狀 →
  dialog / dispatch / controller readers 不動。live setup progress **不再 splice 進 DeviceSnapshot**，改由
  `operation.progress`（ProgressService, by operation_id）單獨查詢。
- `device:<name>` 的 version bump 從 `_emit_device_changed`（縮成純 emit signal）**移進 State mutator**。
- in-flight 暫態（`_active_lease`/`_active_name`/`_active_prior`）綁 worker/gate 生命週期、不可序列化，
  **留 DeviceService**；`_active_prior` 型別改 `DeviceState`。
- 跨物件不變式：device 在 `GlobalDeviceManager` ⟺ State status 是 live 狀態（CONNECTED/SETTING_*）。

### 3. Persistence 為 State 投影（宣告式，取代增量）

- 宣告式：從 State 讀 `remember=True` 的集合、整份覆寫 remembered 集（保留其他欄位）。無 Legacy 兼容。
  - **diff-guard**：DEVICE_CHANGED 每次 transition 都發（含暫態 CONNECTING/SETTING_UP），但投影只依賴
    remember+identity → 只在 remembered 集真的變了才重算寫出。
  - 刪 `remember_device`/`forget_device`。

  > 註：本 §3 描述的磁碟持久化機制（曾由 `StartupPersistenceService` + 增量
  > `add_device`/`remove_device` 承載、`StartupService` 訂閱 `DEVICE_CHANGED` 主動投影到磁碟）已被
  > ADR-0015（Phase 126 PersistenceCaretaker）取代：那些類名/方法名不再存在。`StartupService` 現為
  > stateless、無磁碟 I/O、無 DEVICE_CHANGED 訂閱；磁碟持久化由 `PersistenceCaretaker` 在關閉 flush 時
  > 經 `capture_startup` 投影 remembered device 集寫出。下方依賴圖描述的是當時的事件投影路徑。
- Controller 三 handler 變 **UI-only**（只 `show_status_message`）；`forget_device` 只
  `dev_svc.forget_device`（remove_device → DEVICE_CHANGED → 投影）。

依賴方向（無環）：

```
DeviceService ──寫──▶ State ──emit──▶ EventBus ──DEVICE_CHANGED──▶ StartupService
                                                                      ├─讀─▶ State (remembered 集)
                                                                      └─寫─▶ StartupPersistenceService (純資料層)
StartupService ──restore_devices──▶ DeviceService  (既有邊界，不變；restore 不 emit → 不回投影)
```

## 一般化原則（新增，供後續沿用）

- **version bump = 狀態真的變了，不含「值未變的快取同步」。** 讀取衍生的快取更新:值未變則不 bump;讀到值
  真的變了（外部來源）則 bump + emit changed 事件。
- **persistence 是 State 的投影**：用「State 變了 → 重算整份寫出」的宣告式模型，取代「事件帶 payload → 增量
  改 persistence」。投影者訂閱資源 owner 的 changed 事件、從 State derive、diff-guard、log-and-swallow。

## 否決 / 未做

- driver registry（`GlobalDeviceManager`）本身的 owner 重設計：本 phase 維持 driver 留 DeviceService、
  只下放狀態。
- in-flight 暫態進 State：綁 worker 生命週期、不可序列化，留 service。

## 實作

gui2：Phase 96（`feaa92a4`）/ 97（`2b6ca032`）/ 98（`0f80a83c`）。
