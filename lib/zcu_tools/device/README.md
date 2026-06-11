# Device Note for `zcu_tools/device`

**Last updated:** 2026-06-11（manager 鎖縮小至 registry dict；I/O 由 per-instance 鎖序列化）

這份筆記整理 `lib/zcu_tools/device` 的設計：以 VISA（pyvisa）為底層，抽出 `BaseDevice` + `BaseDeviceInfo` 的通用契約，再由 `GlobalDeviceManager` 做 process-wide 單例管理。目前已實作 `YOKOGS200`（電流/電壓源）、`RohdeSchwarzSGS100A`（微波訊號源）與 `FakeDevice`（mock 測試）。

---

## 架構一覽（一句話版）

`BaseDevice` 封裝 pyvisa session；各硬體子類負責 SCPI 字串；`BaseDeviceInfo`（Pydantic model，繼承 `ConfigBase`）作為 setup/readback 的序列化結構；`DeviceInfo` 是 Union TypeAlias 匯總所有已知子類型；`GlobalDeviceManager` 以名稱為 key 管理已連線的裝置實例，方便實驗 config（YAML/JSON）直接驅動硬體。

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
GlobalDeviceManager (classmethods only, module-level registry)

DeviceInfo = Union[YOKOGS200Info, RohdeSchwarzSGS100AInfo, FakeDeviceInfo]  # in __init__.py
```

---

## `BaseDeviceInfo`（`base.py:15`）

Pydantic model，繼承 `ConfigBase`，所有子類型共用的基底欄位：

| 欄位 | 必填 | 說明 |
| ---- | ---- | ---- |
| `type` | ✔ | 子類名稱（用於 `setup()` sanity check 與序列化分辨） |
| `address` | ✔ | VISA resource string（e.g. `TCPIP::...::INSTR`） |
| `label` | ✘ | 選填顯示名（`Optional[str] = None`） |

子類透過 `class XxxInfo(BaseDeviceInfo)` 繼承，`type: Literal["Xxx"] = "Xxx"` 鎖死型別。

`with_updates(**kwargs)`：拒絕修改 `type` / `address`（protected fields），確保 immutable identity；其他欄位透過 Pydantic `model_validate` 更新。

**`DeviceInfo` TypeAlias**（`__init__.py:11`）：`Union[YOKOGS200Info, RohdeSchwarzSGS100AInfo, FakeDeviceInfo]`，供 `GlobalDeviceManager` API 型別標注用。

---

## `BaseDevice`（`base.py:37`）

職責：

- `info_model: type[BaseDeviceInfo] = BaseDeviceInfo`：class-level 屬性，子類覆寫為對應 Info 型別（用於 `setup()` 的 model_validate）。
- `__init__(address, rm)`：建立 per-instance `_lock`（`threading.RLock`，見「並發與鎖」）；用 pyvisa `ResourceManager` 開 session；強制 `read/write_termination = "\n"`；立即呼叫 `connect_message()`（發 `*IDN?` 並印出）。連線失敗 `pyvisa.Error` 直接 re-raise。
- Helper：`write(cmd)`、`query(cmd)`（已 `strip()`）、`close()`、`connect_message()`。
- Abstract：`_setup(cfg: BaseDeviceInfo, *, progress, stop_event)`、`get_info() -> BaseDeviceInfo`。
- `setup(cfg: BaseDeviceInfo, *, progress, stop_event)`：非抽象 wrapper，fail-fast 流程：
  1. 檢查 `cfg.address == self.address`（否則 `RuntimeError`）。
  2. 檢查 `isinstance(cfg, self.info_model)`（否則 `RuntimeError`）。
  3. Non-blocking acquire `self._lock`；若鎖已被另一執行緒持有 → 立即 `raise DeviceBusyError`（含 class 名與 address）。
  4. 成功取鎖 → try/finally 包住 `self._setup(...)` → release。

**協作式取消（stop_event）**：`setup()` 接受 `Optional[threading.Event]`，透傳給 `_setup`。子類 ramp loop 每步前 check `stop_event.is_set()`，若已設置則 `break`（設備停在中途值，這是預期行為）。SGS100A 無 ramp，簽名對齊但不使用。

**契約重點**：`setup()` 的地址/型別檢查由 base 保證（address 先、`isinstance` 後）；子類 `_setup` 收到的 `cfg` 已是 `self.info_model` 的實例，不需再做 `typeguard.check_type`。

---

## 並發與鎖（per-instance 執行鎖）

每個 device 實例持有一把 **per-instance `threading.RLock`**（`BaseDevice.__init__` 建立；`FakeDevice` 因覆寫 `__init__` 不走 `super().__init__()`，自行於建構時建立同一把鎖）。所有**公開操作**都在這把鎖內執行，確保同一實例的操作彼此**序列化**。

**不變式**：對同一 device 實例的任兩個公開操作不會交錯——一個操作（含整段序列）要嘛完整跑完，要嘛還沒開始；其他 thread 看不到操作的中間狀態。所有公開入口（包括 `output_on` 這類 thin wrapper 的重入）**一律持鎖**，是刻意的統一不變式，避免逐案判斷漏鎖。

**短操作（`get_info`/`query`/`write`/`set_*`/`output_*` 等）**：阻塞等待鎖（blocking acquire），排隊後安全執行。

**`setup()` — fail-fast 語義**：`setup()` 使用 **non-blocking acquire**。若另一執行緒正持鎖操作（例如 ramp 進行中），`setup()` 立即拋出 `DeviceBusyError`（`RuntimeError` 子類），**不排隊、不等待**。這是刻意設計：setup 是長時間操作，靜默排隊難以察覺，快速失敗讓呼叫端有機會向用戶回報衝突。RLock 語義保證**同一執行緒**的嵌套呼叫（`_setup → set_current → output_on`）重入成功，不受影響。

**`is_busy() -> bool`**：non-blocking try-acquire/立即 release 的狀態探測，回傳 True 表示另一執行緒正持鎖。注意兩點：(a) 這是 TOCTOU 快照，僅供顯示/診斷，**不可**用於「先查再做」控制流（控制流靠捕捉 `DeviceBusyError`）；(b) 持鎖執行緒自己呼叫 `is_busy()` 會回 False（RLock 重入成功）。

**鎖粒度**：鎖蓋的是**整個操作序列**，不是單次 I/O——

- YOKOGS200 的 smart ramp 迴圈（逐點 `np.linspace` 下發）整段在鎖內：`set_voltage`/`set_current` 從持鎖到 ramp 完成，期間 `get_info()`/另一個 `setup()` 不會插入而讀到 ramp 中途值。
- `set_frequency`/`set_power`（write 後 read-back 確認）的 write+read 兩步在同一臨界區，避免並發 setter 在兩步之間改值。
- base `setup()` 在呼叫 `_setup` 前取鎖，覆蓋子類整段 `_setup`。

**為何用 RLock（不可用 plain Lock）**：公開入口在同一 thread 上**巢狀**呼叫彼此——`setup() → _setup() → set_current()/set_voltage() → output_on() → set_output() → write()`、以及 `get_voltage()/get_info() → get_mode() → query()`。plain Lock 會在第二層自我死鎖；RLock 允許同 thread 重入。

**鎖覆蓋範圍**：覆蓋每個子類的全部公開出口（不只 `setup`/`get_info`）。私有 helper（`_set_voltage_smart`、`_set_value_smart`、`_get_level` 等）**不自包鎖**，因為它們只從已持鎖的公開方法呼叫，再包一層只是無謂重入。

**呼叫端行為**：GUI `gui/session/services/device.py` 的 worker 呼叫 `driver.setup()`；若拋出 `DeviceBusyError`，錯誤訊息（含 device 名與 address）直接透過 `_on_setup_failed` 呈現給用戶，不做 retry/swallow。`device/manager.py` 的 `setup_devices` 呼叫路徑同理，`DeviceBusyError` 會直接炸掉實驗（預期行為）。

**manager 鎖的職責邊界**：`GlobalDeviceManager._lock` 只守 **registry dict**（哪個名字對應哪個實例）。I/O 操作（`setup()`、`get_info()`）在鎖外執行——即使 `setup_devices` 在鎖外對 A 跑 ramp，對 B 的 `get_info("B")` 可同時進行而不阻塞。兩條 thread 用同一名字 `get_device("flux")` 取得**同一個**實例後，各自呼叫 `setup`/`get_info` 的序列化由 per-instance 鎖負責，manager 鎖不介入。

**持鎖期間禁止 GUI 回呼**：持鎖區段內不得 emit Qt signal 或觸發 GUI 回呼（現狀本來就沒有，維持），以免鎖內等待主執行緒造成跨鎖等待。

---

## `GlobalDeviceManager`（`manager.py:14`）

Process-wide registry（class-level `_devices` dict）：

| API | 用途 |
| --- | ---- |
| `register_device(name, device)` | 註冊；重名會 warn 並覆蓋 |
| `drop_device(name, ignore_error=False)` | 移除 |
| `get_device(name)` | 取得；找不到 `ValueError` |
| `get_all_devices()` | 回傳整個 dict |
| `setup_devices(dev_cfg: Mapping[str, DeviceInfo], *, progress=True)` | 批次對已註冊裝置呼叫 `setup(cfg)` |
| `get_info(name)` | 取得單一 device 的 `get_info()` 結果；找不到 `ValueError` |
| `get_all_info()` | 把每個 device 的 `get_info()` 收成 dict，方便存檔 |

**使用慣例**：在 notebook / 實驗腳本啟動時一次 `register_device`，之後以名稱（如 `"flux"`, `"qubit_lo"`）在任何地方取用。`setup_devices` 通常接受從 YAML / JSON 讀出的 config block。

**Thread safety（鎖分層）**：class 內含 `_lock: threading.RLock`，職責是保護 **registry dict**（`_devices`）的讀寫——僅此而已。I/O 操作（`device.setup()` / `device.get_info()`）全部在 manager 鎖**外**執行，由各 device 的 per-instance `_lock` 序列化。這樣，A 裝置正在跑長時間 ramp 時，對 B 裝置的 `get_info()` 調用不會被 A 的操作阻塞。

**`setup_devices` 的兩段式流程**：
1. **鎖內驗證**：先在 manager 鎖內一次性驗完 `dev_cfg` 裡所有 name 是否已註冊，並取出 instance 引用快照。任一 name 不存在立即 `ValueError`，整批都不執行（fast-fail）。
2. **鎖外執行**：對快照逐一呼叫 `device.setup(cfg, progress=progress)`。期間 manager 鎖已釋放；busy device 會 raise `DeviceBusyError`（fail-fast，不吞）。

**`get_info` / `get_all_info`**：在 manager 鎖內僅做 instance 解析（複用 `get_device` / `get_all_devices`），鎖外才呼叫各裝置的 `get_info()`。`get_all_devices` 回傳 dict shallow copy，外部迭代不 race 於並發的 register/drop。

**注意**：`setup_devices` 假設每個 name 都已註冊；若 cfg 內有未註冊的 name 會 `ValueError`——呼叫端要先 register 再 setup。

---

## 子類實作

### `YOKOGS200`（`yoko.py:34`）— DC 電流／電壓源

**Info 欄位**：`output ∈ {on, off} = "off"`、`mode ∈ {voltage, current} = "voltage"`、`value: float = 0.0`、`rampstep: float = DEFAULT_RAMPSTEP["voltage"]`。

**SCPI 對應**：

- 輸出：`:OUTPut?` / `:OUTPut {0|1}`
- 模式：`:SOURce:FUNCtion?` / `:SOURce:FUNCtion {VOLT|CURR}`
- 位準：`:SOURce:LEVel?` / `:SOURce:LEVel:AUTO {value:.8f}`

**安全上限（寫死）**：電壓 `|V| ≤ 7 V`（`_check_voltage`）、電流 `|I| ≤ 7 mA`（`_check_current`）。超過會 `RuntimeError`。

**Smart ramp**：`set_voltage` / `set_current` 不直接跳值，而是以 `10 * _rampstep` 為步長在 `np.linspace` 上逐點下發，步間 sleep `_rampinterval = 0.01s`。`DEFAULT_RAMPSTEP = {voltage: 1e-4 V, current: 1e-7 A}`。`progress=True` 時跑 make_pbar。`set_voltage` / `set_current` 執行前會先呼叫 `self.output_on()`（自動開啟輸出）。兩者均接受 `stop_event: Optional[threading.Event]`，每步 sleep 前 check `stop_event.is_set()` 協作中止。

**模式切換**：`set_mode(mode, force=False, rampstep=None)` 若當前 level 非零會擋下來，需 `force=True`。切完自動換 `_rampstep` 為新模式的預設（或呼叫端給的）。

**`_setup`**（base 已完成地址/型別檢查，不需 `typeguard.check_type`）：

1. 若裝置 output 為 off 而 cfg 要 on → `warnings.warn`（不自動開，怕暴衝）。
2. **不自動切模式**：cfg mode 與當前不一致直接 `RuntimeError`，明示「切模式要手動先歸零」。
3. 更新 `self._rampstep = cfg.rampstep`。
4. 照 mode 呼叫 `set_voltage` / `set_current`（smart ramp，透傳 `stop_event`）。

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

- **`__init__(fast_mode=False)` 不接受 `rm` 參數**（直接 `self.address = "none"`，跳過 pyvisa session）。`fast_mode=True` 跳過 ramp 的 `time.sleep`（加速測試）。
- `FakeDeviceInfo` 欄位：`type: Literal["FakeDevice"] = "FakeDevice"`、`output: Literal["on","off"] = "off"`、`value: float = 0.0`、`rampstep: float`。
- 提供 `get_output/set_output/output_on/output_off` 與 `get_value/set_value` 方法（記憶體狀態操作）。
- `_set_value_smart(value, progress, stop_event=None)`：以 `np.linspace` 逐步 ramp + make_pbar，每步前 check `stop_event.is_set()`，協作中止。
- `_setup(cfg, *, progress, stop_event)` 呼叫 `set_output` + 更新 `_rampstep` + `_set_value_smart`（透傳 `stop_event`）。
- `get_info()` 回傳當前記憶體狀態。

這個類別設計目標是「可被 `GlobalDeviceManager` 註冊」，模擬完整 ramp 行為（含 pbar + 協作取消）。

---

## 設計要點對照

| 特性 | `YOKOGS200` | `RohdeSchwarzSGS100A` | `FakeDevice` |
| ---- | ----------- | --------------------- | ------------ |
| Smart ramp | ✔（防止突變） | ✘ | ✔（模擬） |
| 協作取消（stop_event） | ✔ | 簽名對齊，不使用 | ✔ |
| 安全上限硬編碼 | ✔（7 V / 7 mA） | 頻率/功率範圍檢查 | ✘ |
| 自動切模式 | ✘（明確 raise） | N/A | N/A |
| output 自動開關 | 僅 warn | 依 cfg 直接設 | 依 cfg 直接設 |
| make_pbar 進度 | ✔（ramp 長時用） | ✘ | ✔ |
| fast_mode 跳 sleep | ✘ | ✘ | ✔（測試用） |

---

## 使用流程範例

```python
import pyvisa
from zcu_tools.device import GlobalDeviceManager
from zcu_tools.device.yoko import YOKOGS200, YOKOGS200Info
from zcu_tools.device.sgs100a import RohdeSchwarzSGS100A, RohdeSchwarzSGS100AInfo

rm = pyvisa.ResourceManager()
flux = YOKOGS200("USB0::...::INSTR", rm)
lo   = RohdeSchwarzSGS100A("TCPIP::...::INSTR", rm)

GlobalDeviceManager.register_device("flux", flux)
GlobalDeviceManager.register_device("qubit_lo", lo)

# 用 Pydantic model 建立 cfg（也可從 YAML/JSON dict 用 model_validate）：
dev_cfg = {
    "flux":     YOKOGS200Info(address="USB0::...::INSTR", output="on",
                              mode="current", value=0.0, rampstep=1e-7),
    "qubit_lo": RohdeSchwarzSGS100AInfo(address="TCPIP::...::INSTR", output="on",
                                        IQ="on", freq_Hz=5.0e9, power_dBm=10.0),
}
GlobalDeviceManager.setup_devices(dev_cfg, progress=True)

# 存檔當前狀態：
snapshot = GlobalDeviceManager.get_all_info()
```

---

## 新增裝置的步驟（checklist）

1. 在 `device/` 下新增 `xxx.py`。
2. 定義 `XxxInfo(BaseDeviceInfo)`，欄位全列、`type: Literal["Xxx"] = "Xxx"`。
3. 繼承 `BaseDevice`：
   - 設 `info_model = XxxInfo`。
   - SCPI getter/setter（`write` / `query`）。
   - 必要時加範圍檢查、smart ramp、安全限制。
   - 實作 `_setup(cfg: XxxInfo, *, progress: bool = True, stop_event: Optional[threading.Event] = None)` 與 `get_info() -> XxxInfo`。
   - 若有 ramp loop：每步前 check `stop_event and stop_event.is_set()`，中止時 `break`。
   - **不需要** `typeguard.check_type`；base `setup()` 已完成型別/地址檢查。
4. `info_model = XxxInfo` 必須設定正確（`BaseDevice.setup` 用 `isinstance(cfg, self.info_model)` 做型別 sanity check）；`XxxInfo.type` literal 供序列化辨別用。
5. 在 `__init__.py` 的 `DeviceInfo` Union 加入 `XxxInfo`，並加進 `__all__`。
6. 在使用端 `GlobalDeviceManager.register_device(name, instance)`。

---

## 已知侷限 / 維護提醒

- `BaseDevice.__init__` 直接 import `pyvisa`（lazy import）；執行環境需先裝 `pyvisa` 與對應 VISA backend（NI-VISA / pyvisa-py）。`FakeDevice` 例外，不需要 pyvisa。
- `GlobalDeviceManager._devices` 是 class-level state，測試或多實驗混用時要注意汙染；必要時手動 `drop_device`。
- `YOKOGS200.set_voltage/set_current` 會**自動呼叫 `output_on()`**，若硬體尚未準備好可能造成問題。
- 安全上限是**硬編碼**在程式內，若更換樣品 / 線路需直接改 `_check_voltage` / `_check_current`。
- `RohdeSchwarzSGS100A` 沒有 ramp，改功率是瞬態——必要時呼叫端自行步進。
- `BaseDeviceInfo.with_updates()` 保護 `type` 和 `address` 欄位不可修改；嘗試修改會 `ValueError`。

---

## 原始碼出處

- `lib/zcu_tools/device/base.py` — `BaseDeviceInfo`（Pydantic ConfigBase）, `BaseDevice`
- `lib/zcu_tools/device/manager.py` — `GlobalDeviceManager`
- `lib/zcu_tools/device/__init__.py` — `DeviceInfo` Union TypeAlias 與對外匯出
- `lib/zcu_tools/device/yoko.py` — `YOKOGS200Info`, `YOKOGS200`
- `lib/zcu_tools/device/sgs100a.py` — `RohdeSchwarzSGS100AInfo`, `RohdeSchwarzSGS100A`
- `lib/zcu_tools/device/fake.py` — `FakeDeviceInfo`, `FakeDevice`
