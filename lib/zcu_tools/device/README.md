# Device Note for `zcu_tools/device`

**Last updated:** 2026-10-27

**讀者：** framework-dev。儀器 driver、DeviceManager、並發與安全限制速查。

這份筆記整理 `lib/zcu_tools/device` 的設計：以 VISA（pyvisa）為底層，抽出 `BaseDevice` + `BaseDeviceInfo` 的通用契約，再由 `DeviceManager` 管理 caller 明確建立的 registry。內建裝置包含 `YOKOGS200`（電流/電壓源）、`RohdeSchwarzSGS100A`（微波訊號源）與 `FakeDevice`（mock 測試）。

---

## 目錄

- [架構一覽（一句話版）](#架構一覽一句話版)
- [DeviceManager 單名稱登記](#devicemanager-單名稱登記)
- [活動觀察](#活動觀察)
- [實驗控制與 acquisition snapshot](#實驗控制與-acquisition-snapshot)
- [`BaseDeviceInfo`（`base.py:15`）](#basedeviceinfobasepy15)
- [`BaseDevice`（`base.py:37`）](#basedevicebasepy37)
- [並發與鎖（per-instance `op_lock` / `io_lock`）](#並發與鎖per-instance-op_lock--io_lock)
- [`DeviceManager`（`manager.py:14`）](#devicemanagermanagerpy14)
- [子類實作](#子類實作)
- [設計要點對照](#設計要點對照)
- [使用流程範例](#使用流程範例)
- [新增裝置的步驟（checklist）](#新增裝置的步驟checklist)
- [已知侷限 / 維護提醒](#已知侷限--維護提醒)
- [原始碼出處](#原始碼出處)

## 架構一覽（一句話版）

`BaseDevice` 封裝 pyvisa session；各硬體子類負責 SCPI 字串；`BaseDeviceInfo`（Pydantic model，繼承 `ConfigBase`）作為 setup/readback 的序列化結構；`DeviceInfo` 是 Union TypeAlias 匯總所有已知子類型；`DeviceManager` 以名稱為 key 管理已連線的裝置實例，供 caller 明確設定、讀回與釋放資源。

```text
BaseDeviceInfo (Pydantic ConfigBase)
      ↑ subclass
XxxInfo(BaseDeviceInfo)  ← cfg in/out
      ↑ info_model ref
BaseDevice (ABC, wraps pyvisa session)
      ↑
 ┌────┴──────────────────┬─────────┐
 YOKOGS200    RohdeSchwarzSGS100A  FakeDevice
      ↑ register_device
DeviceManager (instance-owned registry and close claims)

DeviceInfo = Union[
    YOKOGS200Info,
    RohdeSchwarzSGS100AInfo,
    FakeDeviceInfo,
    AnritsuMG3692Info,
]  # in __init__.py
```

---

## DeviceManager 單名稱登記

`register_device` 拒絕含 `.` 的名稱，拋出指出名稱的 `ValueError`，registry 不變且不做儀器 I/O。點號用於 component 路徑與 record 攤平欄名的分隔。其他名稱原樣保留，包括空字串，不查 setup 或連線狀態。

名稱規則由 `cfg_model.validate_device_name` 擁有，與 `DeviceName` 型別驗證共用。`device.manager → cfg_model` 依賴穩定值型別層，符合 ADR-0072。Import contracts C8 允許同層使用，C9 禁止循環，C11 禁止 cfg_model 反向依賴 device；不需要忽略規則。

同一個 driver 在一個 `DeviceManager` 中只接受一個名字。`register_device` 遇到已用其他名字登記的同一實例時拋出 `ValueError`，registry 保留原狀。同名登記仍發出 warning 並覆寫該名稱，可以登記原 driver 或另一個 driver。不同名稱可以各自登記不同 driver。

Manager close 尚未完成時，`register_device` 拒絕登記該 driver，拋出 `DeviceCloseInProgressError`。此限制持續到 close 完成，即使原名稱已被 drop 或被另一個 driver 覆寫。

## 活動觀察

Driver 在 `device_operation` 內記錄自己的兩項活動資訊。`DeviceActivity.in_operation` 初始為 `False`，最外層操作開始後為 `True`，最外層操作結束時才清除。`last_finished_monotonic` 初始為 `None`，最外層操作成功或失敗結束時更新為 `time.monotonic()` 的秒數。Busy rejection 不改活動資訊。

`DeviceManager.activity_snapshot()` 回傳目前登記名字到 `DeviceActivity` 的 dict，每台一筆。`BaseDevice.activity` 提供同一種只讀觀察。快照直接取樣記憶體中的當前欄位，獨立於硬體 I/O、operation lock 與 callback。快照只保存當下值。回傳 dict 與 driver 分離，值為唯讀。

任意 thread 都可讀取活動快照，包括正在操作的 thread。Registry 名單在 manager 鎖內複製，活動欄位在鎖外分別取樣，不保證跨欄位或跨 driver 原子一致。時間只與同一 process 的 monotonic clock 比較，不當作 wall-clock 時間。GUI 定時讀取這些資訊作顯示，不用活動快照決定 admission、取消或 settlement。

## 實驗控制與 acquisition snapshot

實驗 cfg 自行宣告 `dev`，保存角色到 device 名稱的 mapping。Core 在實際使用時從 `RunContext.devices[name]` 取得 driver，明確設定所需裝置。

`DeviceInfo` 描述設定與讀回結果，driver 提供儀器操作。Flux 與 RF 等實驗角色、單位檢查與操作 helper 由使用者套件擁有。Flux helper 明示 A、V 或 native，以 Info 的 `value_unit` 檢查單位，保留目前 mode、步長與 output。YOKO 與 Fake 的 Info 支援此 flux 設定。Helper 的 owner 與使用入口見 [experiment 的 Device 操作邊界](../experiment/README.md#device-操作邊界)。

GUI 在既有操作 lease 後、core 前，Notebook 在建立 caller-owned `RunContext` 後、core 前，各讀所有 connected drivers 一次。固定 snapshot 進入 `RunMetadata.device_snapshot`，Save 不刷新硬體。`DeviceManager.setup_devices` 與 GUI 手動設定仍是獨立的明確操作。

## `BaseDeviceInfo`（`base.py:15`）

Pydantic model，繼承 `ConfigBase`，所有子類型共用的基底欄位：

| 欄位 | 必填 | 說明 |
| ---- | ---- | ---- |
| `type` | ✔ | 子類名稱（用於 `setup()` sanity check 與序列化分辨） |
| `address` | ✔ | VISA resource string（e.g. `TCPIP::...::INSTR`） |
| `label` | ✘ | 選填顯示名（`Optional[str] = None`） |

子類透過 `class XxxInfo(BaseDeviceInfo)` 繼承，`type: Literal["Xxx"] = "Xxx"` 鎖死型別。

Value-capable info models（目前 `FakeDeviceInfo` / `YOKOGS200Info`）把 `value`
視為 strict real numeric scalar：只接受 `int | float`，不接受 `bool` 或 string
coercion。這個限制在 Pydantic model boundary 先 fail-fast，避免 device cache /
predictor device-value flow 把非 numeric input 當作有效硬體值。

`with_updates(**kwargs)`：拒絕修改 `type` / `address`（protected fields），確保 immutable identity；其他欄位透過 Pydantic `model_validate` 更新。

**`DeviceInfo` TypeAlias**（`__init__.py:12`）匯總所有已知 info model，供 `DeviceManager` API 型別標注用。

---

## `BaseDevice`（`base.py:37`）

職責：

- `info_model: ClassVar[type[BaseDeviceInfo]]`：子類必須在 class body 宣告具體 Info 型別；漏宣告、宣告 `BaseDeviceInfo` 本身、或宣告非 `BaseDeviceInfo` subclass 會在 class definition 時 `TypeError`。
- `__init__(address, rm)`：建立 per-instance `_op_lock`（`threading.RLock`）/ `_io_lock`（`threading.Lock`，見「並發與鎖」）與 logger；透過 protected `_open_session(rm)` 建立 session。預設 `_open_session` 要求非 `None` pyvisa `ResourceManager`；`FakeDevice` override 回傳 `None`，因此不開真 session 但仍取得 base invariants。
- 若 `_open_session()` 回傳 session，base 會強制 `read/write_termination = "\n"` 並呼叫 `connect_message()`（發 `*IDN?` 並寫 logger）。連線失敗 `pyvisa.Error` 以 logger 記錄後 re-raise。
- Helper：`write(cmd)`、`query(cmd)`（已 `strip()`，只持 `_io_lock`）、`close()`、`connect_message()`；無 session 時 `write/query/connect_message/close` fail-fast，除非子類像 `FakeDevice.close()` 明確 override 成安全 no-op。
- Abstract：`_setup(cfg: BaseDeviceInfo, *, progress, stop_event)`、`get_info() -> BaseDeviceInfo`。
- `setup(cfg: BaseDeviceInfo, *, progress, stop_event)`：非抽象 wrapper，fail-fast 流程：
  1. `@device_operation` 以 non-blocking acquire 取得 `_op_lock`；若另一執行緒正在執行 mutating operation → 立即 `raise DeviceBusyError`（含 class 名與 address）。
  2. decorator 只對外層操作寫 logger：started / finished / failed / busy；同 thread nested setter 不重複刷 log。
  3. 檢查 `cfg.address == self.address`（否則 `RuntimeError`）。
  4. 檢查 `isinstance(cfg, self.info_model)`（否則 `RuntimeError`）。
  5. 呼叫 `self._setup(...)`；子類內部的 setter 透過 reentrant `_op_lock` 安全重入。

**協作式取消（stop_event）**：`setup()` 接受 `Optional[threading.Event]`，透傳給 `_setup`。子類 ramp loop 每步前 check `stop_event.is_set()`，若已設置則 `break`（設備停在中途值，這是預期行為）。SGS100A 無 ramp，簽名對齊但不使用。

**契約重點**：`setup()` 的地址/型別檢查由 base 保證（address 先、`isinstance` 後）；子類 `_setup` 收到的 `cfg` 已是 `self.info_model` 的實例，不需再做 `typeguard.check_type`。`info_model` 的存在與有效性在 subclass 定義時先被檢查，runtime `setup()` 仍保留 cfg sanity check。

---

## 並發與鎖（per-instance `op_lock` / `io_lock`）

每個 device 實例持有兩把鎖（`BaseDevice.__init__` 建立；`FakeDevice` 透過 `_open_session()` hook 跳過真 session，但仍走同一個 base init 流程）。

- `_op_lock`：保護 public mutating operation（`setup`、`close`、`set_*`、`output_*`、`IQ_*`）。入口一律用 `@device_operation`，non-blocking acquire，忙碌時 fail-fast。
- `_io_lock`：普通 `threading.Lock`，只保護 base `write` / `query` / `close` 這類單次 session transaction。driver getter / `get_info()` 不手動取得 `_io_lock`；它們只透過每一次 `query()` 間接取得短 mutex。

**不變式**：同一 device 的 mutating operation 彼此互斥；同一 thread 的 nested setter 透過 `_op_lock` 的 RLock 重入。Read-only getter 只在每次 `query()` 取得短 `_io_lock`，因此可在長 ramp 的 sleep 間插入讀取，也可能看到 ramp 中途狀態。GUI device dialog 可在長 setup/ramp 期間讀回資料。

**mutating operation — fail-fast 語義**：所有套 `@device_operation` 的 public mutating operation 使用 **non-blocking acquire**。若另一執行緒正持 `_op_lock`（例如 ramp 進行中），立即拋出 `DeviceBusyError`，由 caller 處理這次衝突。setup 是長時間操作，靜默排隊難以察覺，快速失敗讓呼叫端有機會向用戶回報衝突。

**`is_busy() -> bool`**：non-blocking try-acquire `_op_lock` / 立即 release 的狀態探測，回傳 True 表示另一執行緒正在執行 mutating operation。注意兩點：(a) 這是 TOCTOU 快照，僅供顯示/診斷，**不可**用於「先查再做」控制流（控制流靠捕捉 `DeviceBusyError`）；(b) 持 `_op_lock` 的執行緒自己呼叫 `is_busy()` 會回 False（RLock 重入成功）。

**鎖粒度**：

- YOKOGS200 / FakeDevice 的 smart ramp 整段持 `_op_lock`，所以第二個 setter/setup 會 `DeviceBusyError`；YOKOGS200 每一步的 SCPI `write()` 只短暫持 `_io_lock`，sleep 不持 `_io_lock`，因此 `get_info()` 可在 ramp 中讀取中途值。
- `RohdeSchwarzSGS100A.set_frequency` / `set_power` 的 write 和 read-back 各自透過 `write()` / `query()` 取得短 `_io_lock`；`_op_lock` 防止另一個 setter 同時改值，getter 插入讀取是允許的。
- base `setup()` 在呼叫 `_setup` 前取得 `_op_lock`，覆蓋子類整段 `_setup`。

**為何用 RLock（不可用 plain Lock）**：公開入口在同一 thread 上**巢狀**呼叫彼此——`setup() → _setup() → set_current()/set_voltage() → set_output() → write()`、以及 `get_voltage()/get_info() → get_mode() → query()`。plain Lock 會在第二層自我死鎖；RLock 允許同 thread 重入。

**鎖覆蓋範圍**：`@device_operation` 覆蓋每個子類的 public mutating 出口。Read-only getter 透過 `query()` 取得短 `_io_lock`。私有 helper（`_set_voltage_smart`、`_set_value_smart`、`_get_level` 等）沿用公開方法已持有的 `_op_lock`。

**呼叫端行為**：GUI `gui/session/services/device.py` 的 worker 呼叫 `driver.setup()`；若拋出 `DeviceBusyError`，錯誤訊息（含 device 名與 address）直接透過 `_on_setup_failed` 呈現給用戶，不做 retry/swallow。`device/manager.py` 的 `setup_devices` 呼叫路徑同理，`DeviceBusyError` 會直接炸掉實驗（預期行為）。

**manager 鎖的職責邊界**：`DeviceManager._lock` 只守 **registry dict**（哪個名字對應哪個實例）。device 操作在 manager 鎖外執行。兩條 thread 用同一名字 `get_device("flux")` 取得**同一個**實例後，mutating operation 由該 instance 的 `_op_lock` 序列化；getter 只會在實際 `query()` 時等待 `_io_lock` 的短 I/O transaction。

**GUI 回呼順序**：釋放鎖後才 emit Qt signal 或觸發 GUI 回呼，避免鎖內等待主執行緒造成跨鎖等待。

---

## `DeviceManager`（`manager.py:14`）

每個 manager 擁有自己的 registry、lock 與 close claims。共享 driver 的 callers 必須共享同一 manager，不由不同 manager 重複取得關閉權。

| API | 用途 |
| --- | ---- |
| `register_device(name, device)` | 登記 `BaseDevice` 實例；拒絕含點號名稱，同一 driver 只接受一個名字，拒絕正在 close 的 driver，同名仍 warn 並覆寫 |
| `drop_device(name, *, ignore_error=False)` | 移除 registry entry（不 close） |
| `get_device(name)` | 取得；找不到 `ValueError` |
| `get_all_devices()` | 回傳整個 dict |
| `activity_snapshot()` | 回傳每個已登記名稱的唯讀活動值，不讀硬體、不等待長操作 |
| `setup_devices(dev_cfg: Mapping[str, DeviceInfo], *, progress=True, cancel_signal=None)` | 批次對已註冊裝置呼叫 `setup(cfg)` |
| `get_info(name)` | 取得單一 device 的 `get_info()` 結果；找不到 `ValueError` |
| `get_all_info()` | 把每個 device 的 `get_info()` 收成 dict，方便存檔 |
| `close_device(name, *, ignore_missing=False)` | close 單一 device，成功後移除仍指向該 driver 的登記 |
| `close_all_devices()` | close 所有 registered devices，聚合 named failures |

**registry-owned disconnect（`close_device` / `close_all_devices`）**：manager 以
`id(device)` 為 key 維護 private in-flight close claims（claim 持有 strong reference，防止
identity 被回收重複使用）；registry lock 只保護 lookup/claim/cleanup，actual
`device.close()` 永遠在 lock 外執行。語義：

- 同一 identity 已被任何 manager close API claim 時，第二個呼叫者立即收到
  `DeviceCloseInProgressError`（fail-fast，不等待、不重複 close）。
- close 成功後移除仍指向該 driver 的原名稱登記。同名不同 replacement 保留。
  Batch 開始後才登記的不同 device 不屬正在進行的 batch。
- ordinary failure 包裝為 `DeviceCloseFailure(names, cause)`，names 是該筆登記名稱的
  單元素 tuple。Manager release claim 並保留 entry 供重試。`BaseException` 不包裝，
  直接傳播，但一樣 release claim。
- `close_all_devices()` 依 registry snapshot 嘗試 close 所有可 claim drivers。
  Ordinary failures 與 in-progress drivers 不阻斷其餘 entries，in-progress 以
  `DeviceCloseInProgressError` 記錄該筆名稱。Manager 將這些錯誤與 named failures
  一起以 built-in `ExceptionGroup` 拋出。Empty registry 是 no-op。
- close 只 disconnect session，絕不執行 RF/current/voltage state mutation；也不會取代
  `drop_device()` 或 GUI disconnect 語義。`BaseDevice.close()` 仍按 driver 自己的 `_op_lock` /
  `_io_lock` 同步；registry claim 不取代 driver 鎖。
- failure 保留 registry entries 供明確處置，並不保證部分 close 後的 session 可繼續使用、
  再試一定成功或所有資源都已斷線。factory owner 要在所管理的 device 成功 disconnect 後
  才關閉 ResourceManager；disconnect 不等於硬體已處於安全狀態。

**使用慣例**：在 notebook / 實驗腳本啟動時一次 `register_device`，之後以名稱（如 `"flux"`, `"qubit_lo"`）透過同一 manager 取用。`register_device` 只接受 `BaseDevice` instance，duck object 或裸 mock 會在入口 fail-fast。`setup_devices` 通常接受從 YAML / JSON 讀出的 config block。

**Thread safety（鎖分層）**：每個 instance 內含 `_lock: threading.RLock`，職責是保護 **registry dict**（`_devices`）的讀寫——僅此而已。`device.setup()` / `device.get_info()` 全部在 manager 鎖**外**執行。各 device 自己用 `_op_lock` 管 mutating operation、`_io_lock` 管單次 session transaction；A 裝置正在跑長時間 ramp 時，對 B 裝置的 `get_info()` 不會被 A 阻塞，對 A 自己的 `get_info()` 也只等待短 I/O 而不等待整段 ramp。

**`setup_devices` 的兩段式流程**：
1. **鎖內驗證**：先在 manager 鎖內一次性驗完 `dev_cfg` 裡所有 name 是否已註冊，並取出 instance 引用快照。任一 name 不存在立即 `ValueError`，整批都不執行（fast-fail）。
2. **鎖外執行**：先解析取消訊號，然後對快照逐一呼叫
   `device.setup(cfg, progress=progress, stop_event=resolved_cancel_signal)`。期間
   manager 鎖已釋放；busy device 會 raise `DeviceBusyError`（fail-fast，不吞）。

**顯式取消**：`setup_devices` 只使用 caller 傳入的 `cancel_signal`，不讀 ambient scope。整批名稱驗證先於取消短路；Event 已 set 時不再開始下一台 setup。

**`get_info` / `get_all_info`**：在 manager 鎖內僅做 instance 解析（複用 `get_device` / `get_all_devices`），鎖外才呼叫各裝置的 `get_info()`。`get_all_devices` 回傳 dict shallow copy，外部迭代不 race 於並發的 register/drop。

**注意**：`setup_devices` 假設每個 name 都已註冊；若 cfg 內有未註冊的 name 會 `ValueError`——呼叫端要先 register 再 setup。

---

## 子類實作

### `YOKOGS200`（`yoko.py:34`）— DC 電流／電壓源

**Info 欄位**：`output ∈ {on, off} = "off"`、必填 `mode ∈ {voltage, current}`、`value: float = 0.0`。`value_unit ∈ {A, V}` 是儲存欄位；省略時由 Info 依 mode 產生，current 為 A，voltage 為 V。顯式矛盾單位或缺 mode 在 model boundary 拋 `ValidationError`。Model dump 與完整 cfg round-trip 保留 `value_unit`。`rampstep` 依 mode 預設為 1e-3 V 或 1e-6 A，必須為正且有限。上限都是 module 常數，不可由 setup 設定：步長上限 `MAX_RAMPSTEP` 為 1e-2 V 與 1e-5 A，輸出絕對值上限 `MAX_VOLTAGE`＝20 V 與 `MAX_CURRENT`＝20 mA。

**SCPI 對應**：

- 輸出：`:OUTPut?` / `:OUTPut {0|1}`
- 模式：`:SOURce:FUNCtion?` / `:SOURce:FUNCtion {VOLT|CURR}`
- 位準：`:SOURce:LEVel?` / `:SOURce:LEVel:AUTO {value:.8f}`

**安全上限**：`YOKOGS200Info` 依 mode 驗證 `value` 不超過設定上限；GUI、RPC 更新與 saved setup 共用這項驗證。Driver 在 setup 採用上限，後續直接 setter 也遵守，超過會 `RuntimeError`。上限可依 sample 設定，不代表儀器或線路能力的自動判定。

**Smart ramp**：`set_voltage` / `set_current` 以 `_rampstep` 為實際步長在 `np.linspace` 上逐點下發，步間 sleep `_rampinterval = 0.01s`。`DEFAULT_RAMPSTEP = {voltage: 1e-3 V, current: 1e-6 A}`。線性 stepping、progress 與 `stop_event` check 由 module-private shared ramp helper 處理；mode/output/safety guard 保留在 YOKO public setter 與 direct setter。YOKO ramp 保留起始點也下發一次的硬體 I/O/timing 行為。`progress=True` 時跑 make_pbar。若 output 為 off 且目標值非零，setter 會 `RuntimeError`，不自動開輸出。兩者均接受 `stop_event: Optional[threading.Event]`，每步前 check `stop_event.is_set()` 協作中止。

**模式切換**：`set_mode(mode, rampstep=None, *, force=False)` 若當前 level 非零會擋下來，需 `force=True`。切完自動換 `_rampstep` 為新模式的預設（或呼叫端給的）。

**`_setup`**（base 已完成地址/型別檢查，不需 `typeguard.check_type`）：

1. 若裝置 output 為 off 而 cfg 要 on，發出 `warnings.warn` 並維持 off，避免暴衝。呼叫端須明確開啟 output。
2. cfg mode 與當前不一致時拋出 `RuntimeError`。呼叫端須先手動歸零，再明確切換模式。
3. 採用 cfg 的 rampstep。
4. 照 mode 呼叫 `set_voltage` / `set_current`（smart ramp，透傳 `stop_event`）。

**`get_info`**：output、mode 與 level 各讀一次，Info 依這次 mode 產生並保存單位，不增加硬體查詢。GUI snapshot 讀有效 cached Info 的 `value_unit`，不另推導 mode。

**`get_voltage` / `get_current`**：呼叫 `get_mode()` 檢查 mode 是否符合（否則 raise），然後直接讀 `_get_level()`（即 `:SOURce:LEVel?`），無 re-write 副作用。

---

### `RohdeSchwarzSGS100A`（`sgs100a.py:23`）— 微波 LO / IQ 源

**Info 欄位**：`output`、`IQ`、`freq_Hz`、`power_dBm`。

**SCPI 對應**：

- 輸出：`:OUTPut?` / `:OUTPut {0|1}`
- IQ 調變：`:IQ:STAT?` / `:IQ:STAT {0|1}`
- 頻率：`SOUR:FREQ?` / `:SOUR:FREQ {:.2f}`（範圍 `[1 MHz, 20 GHz]`）
- 功率：`SOUR:POW:POW?` / `:SOUR:POW:POW {:.2f}`（範圍 `[-120, 25] dBm`）

**`_setup`**：純一次性下發四個 setter，**沒有 smart ramp**（LO 切換本來就是瞬態）。

---

### `FakeDevice`（`fake.py`）— mock 測試設備

`FakeDevice` 用於本地流程測試，不需要真實儀器連線：

- **`FakeDevice(*, address=None, fast_mode=False, value_unit="native")` 不接受 `rm` 參數**。顯式 address 從 process 共享的 class 容器取得同一實例。首次建立固定單位與 fast_mode，重取保留 value、output、ramp 與模擬綁定。
- 省略 address 每次建立獨立 anonymous instance，不進容器。它的 Info address 是 `"none"`，與顯式 address `"none"` 的身分不同。
- Class 容器最多保留 1024 個 addressed instances，僅防止無限增長。滿額拒絕新 address，不淘汰，既有 address 仍可取回。`close()` 不移除身分。真實 driver 不受影響，見 ADR-0076。
- `isolated_instances()` 在無工作執行時建立空 scope，退出後還原原容器。測試使用它隔離 process 共享狀態。
- Fake 不開 pyvisa session，但首次建立仍取得 base locks/logger/session invariant。`fast_mode=True` 跳過 ramp 的 `time.sleep`。`value_unit` 為 native、V 或 A，不做數值換算。
- `FakeDeviceInfo` 欄位：`type: Literal["FakeDevice"] = "FakeDevice"`、`output: Literal["on","off"] = "off"`、`value: float = 0.0`、`rampstep: float = 0.01`、`value_unit: Literal["native", "V", "A"] = "native"`。Value 與 rampstep 使用相同固定單位。Setup 單位不同時在任何寫入前拒絕；換單位須重建裝置。Dump 與 GUI snapshot 保留實際單位。RF 裝置的 Hz／dBm 量各有固定單位，不新增 `value_unit`。
- 提供 `get_output/set_output/output_on/output_off` 與 `get_value/set_value` 方法（記憶體狀態操作）。
- `_set_value_smart(value, stop_event=None, *, progress=False)`：以 `_rampstep` 為實際步長，透過同一個 private ramp helper 做 `np.linspace` stepping + make_pbar；Fake 保留跳過起始點的 mock 行為，每步前 check `stop_event.is_set()`，協作中止。
- `_setup(cfg, *, progress, stop_event)` 先驗證固定單位，再呼叫 `set_output` + 更新 `_rampstep` + `_set_value_smart`（透傳 `stop_event`）。
- `get_info()` 回傳當前記憶體狀態。

這個類別設計目標是「可被 `DeviceManager` 註冊」，模擬完整 ramp 行為（含 pbar + 協作取消）。

---

## 設計要點對照

| 特性 | `YOKOGS200` | `RohdeSchwarzSGS100A` | `FakeDevice` |
| ---- | ----------- | --------------------- | ------------ |
| Smart ramp | ✔（防止突變） | ✘ | ✔（模擬） |
| 協作取消（stop_event） | ✔ | 簽名對齊，不使用 | ✔ |
| 安全上限 | Info 可設定（預設 20 V / 20 mA） | 頻率/功率範圍檢查 | ✘ |
| 自動切模式 | ✘（明確 raise） | N/A | N/A |
| output 自動開關 | 僅 warn | 依 cfg 直接設 | 依 cfg 直接設 |
| make_pbar 進度 | ✔（ramp 長時用） | ✘ | ✔ |
| fast_mode 跳 sleep | ✘ | ✘ | ✔（測試用） |

---

## 使用流程範例

```python
import pyvisa
from zcu_tools.device import DeviceManager
from zcu_tools.device.yoko import YOKOGS200, YOKOGS200Info
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo

manager = DeviceManager()
rm = pyvisa.ResourceManager()
flux = YOKOGS200("USB0::...::INSTR", rm)
lo   = RohdeSchwarzSGS100A("TCPIP::...::INSTR", rm)

manager.register_device("flux", flux)
manager.register_device("qubit_lo", lo)

# 用 Pydantic model 建立 cfg（也可從 YAML/JSON dict 用 model_validate）：
dev_cfg = {
    "flux":     YOKOGS200Info(address="USB0::...::INSTR", output="on",
                              mode="current", value=0.0, rampstep=1e-7),
    "qubit_lo": RohdeSchwarzSGS100AInfo(address="TCPIP::...::INSTR", output="on",
                                        IQ="on", freq_Hz=5.0e9, power_dBm=10.0),
}
manager.setup_devices(dev_cfg, progress=True)

# 存檔當前狀態：
snapshot = manager.get_all_info()
```

---

## 新增裝置的步驟（checklist）

1. 在 `device/` 下新增 `xxx.py`。
2. 定義 `XxxInfo(BaseDeviceInfo)`，欄位全列、`type: Literal["Xxx"] = "Xxx"`。
3. 繼承 `BaseDevice`：
   - 設 `info_model = XxxInfo`；漏宣告或宣告錯誤會在 class definition 時 fail-fast。
   - SCPI getter/setter（`write` / `query`）。
   - Public mutating method 加 `@device_operation`；getter 不加 decorator，也不手動包 `_io_lock`，只依靠 `write` / `query` 的短 mutex。
   - 必要時加範圍檢查、smart ramp、安全限制；smart ramp 可重用 module-private helper，但 device-specific mode/output/safety guard 仍留在 driver shell。
   - 實作 `_setup(cfg: XxxInfo, *, progress: bool = True, stop_event: Optional[threading.Event] = None)` 與 `get_info() -> XxxInfo`。
   - 若有 ramp loop：每步前 check `stop_event and stop_event.is_set()`，中止時 `break`。
   - 由 base `setup()` 統一完成型別與地址檢查。
4. `info_model = XxxInfo` 必須設定正確（class definition guard 先檢查，`BaseDevice.setup` 再用 `isinstance(cfg, self.info_model)` 做 runtime sanity check）；`XxxInfo.type` literal 供序列化辨別用。
5. 在 `__init__.py` 的 `DeviceInfo` Union 加入 `XxxInfo`，並加進 `__all__`。
6. 在使用端 `manager.register_device(name, instance)`。

---

## 已知侷限 / 維護提醒

- 預設 `BaseDevice._open_session()` 直接 import `pyvisa`（lazy import）；執行環境需先裝 `pyvisa` 與對應 VISA backend（NI-VISA / pyvisa-py）。`FakeDevice` override `_open_session()`，不需要 pyvisa。
- 建立者明確執行 disconnect，成功後才關閉其 ResourceManager。Manager 的關閉時機由建立者管理。
- `YOKOGS200.set_voltage/set_current` 在 output off 且目標值非零時會拒絕執行；呼叫端要先明確開啟 output。
- YOKO 輸出與步長上限由 setup 設定。預設步長上限為模式預設步長的 10 倍，caller 必須按 sample 與線路條件選擇適用上限。
- `RohdeSchwarzSGS100A` 沒有 ramp，改功率是瞬態——必要時呼叫端自行步進。
- `BaseDeviceInfo.with_updates()` 保護 `type` 和 `address` 欄位不可修改；嘗試修改會 `ValueError`。

---

## 原始碼出處

- `lib/zcu_tools/device/base.py` — `BaseDeviceInfo`（Pydantic ConfigBase）, `BaseDevice`
- `lib/zcu_tools/device/manager.py` — `DeviceManager`
- `lib/zcu_tools/device/__init__.py` — `DeviceInfo` Union TypeAlias 與對外匯出
- `lib/zcu_tools/device/yoko.py` — `YOKOGS200Info`, `YOKOGS200`
- `lib/zcu_tools/device/sgs100a.py` — `RohdeSchwarzSGS100AInfo`, `RohdeSchwarzSGS100A`
- `lib/zcu_tools/device/fake.py` — `FakeDeviceInfo`, `FakeDevice`
- `lib/zcu_tools/device/mg3692.py` — `AnritsuMG3692Info`, `AnritsuMG3692`
