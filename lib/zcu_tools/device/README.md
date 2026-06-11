# Device Note for `zcu_tools/device`

**Last updated:** 2026-06-08

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
- `__init__(address, rm)`：用 pyvisa `ResourceManager` 開 session；強制 `read/write_termination = "\n"`；立即呼叫 `connect_message()`（發 `*IDN?` 並印出）。連線失敗 `pyvisa.Error` 直接 re-raise。
- Helper：`write(cmd)`、`query(cmd)`（已 `strip()`）、`close()`、`connect_message()`。
- Abstract：`_setup(cfg: BaseDeviceInfo, *, progress, stop_event)`、`get_info() -> BaseDeviceInfo`。
- `setup(cfg: BaseDeviceInfo, *, progress, stop_event)`：非抽象 wrapper，流程：
  1. 檢查 `cfg.address == self.address`（否則 `RuntimeError`）。
  2. 檢查 `isinstance(cfg, self.info_model)`（否則 `RuntimeError`）。
  3. 呼叫 `self._setup(cfg, progress=progress, stop_event=stop_event)`。

**協作式取消（stop_event）**：`setup()` 接受 `Optional[threading.Event]`，透傳給 `_setup`。子類 ramp loop 每步前 check `stop_event.is_set()`，若已設置則 `break`（設備停在中途值，這是預期行為）。SGS100A 無 ramp，簽名對齊但不使用。

**契約重點**：`setup()` 的地址/型別檢查由 base 保證（address 先、`isinstance` 後）；子類 `_setup` 收到的 `cfg` 已是 `self.info_model` 的實例，不需再做 `typeguard.check_type`。

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

**Thread safety**：class 內含 `_lock: threading.RLock`，所有 classmethod（`register_device` / `drop_device` / `get_device` / `get_all_devices` / `setup_devices` / `get_info` / `get_all_info`）都在鎖內操作。使用 RLock 以允許同一 thread 在 `setup_devices` 執行期間安全地 re-entry 到 `get_device` 等 API。`get_all_devices` / `get_all_info` 回傳 shallow copy，外部迭代時不會 race 於並發的 register/drop。

**注意**：`setup_devices` 假設每個 name 都已註冊；若 cfg 內有未註冊的 name 會 `ValueError` ——呼叫端要先 register 再 setup。

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
