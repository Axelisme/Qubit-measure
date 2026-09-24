# ADR-0058：registry-owned VISA session disconnect lifecycle

**狀態：** accepted

## Context

`GlobalDeviceManager`（`lib/zcu_tools/device/manager.py`）是 process-wide 單例
registry：class-level `_devices` dict、classmethods，以名稱為 key 管理已連線的
`BaseDevice` 實例；`BaseDevice` 封裝 pyvisa session，`pyvisa.ResourceManager` 是
notebook 層的 session factory。關閉流程若散落 caller，會造成同一 identity 被重複
close、stale aliases 殘留、registry lock 內執行長時間 I/O 阻塞其他 lookup、以及
device session 未斷前 ResourceManager 就被關閉。disconnect 需要單一 registry-owned
contract，且 connection lifecycle 與 hardware-state policy 必須分離（device live
state 由 State 擁有，見 [[0007]]）。

## Decision

1. **close = connection/session disconnect only**。`close_device` / `close_all_devices`
   只斷開 VISA session，絕不執行 RF/current/voltage state mutation（不 output_off、
   不 ramp、不 reset value、不走 device lock semantics）。hardware-state policy 不屬於
   connection lifecycle。

2. **Manager 是 disconnect 的唯一擁有者**。`GlobalDeviceManager.close_device(name, *,
   ignore_missing=False)` 與 `close_all_devices()` classmethods 擁有全部 close 路徑；
   `register_device` / `drop_device` / GUI disconnect 的既有語義不變。

3. **Identity aliases**。registry 以 name→device，多個 name 可指向同一 instance。
   close 成功後移除當下所有仍指向該 identity 的 aliases（含 close 進行中新增的同
   identity alias）；同名不同 replacement（不同 identity）保留，close 後新註冊的
   device 不屬正在進行的 batch。

4. **In-flight claims**。private close claims 以 `id(device)` 為 key，claim 持有
   strong reference，防止 identity 被回收後重用。registry lock（`_lock`）只保護
   lookup / claim / cleanup；actual `device.close()` 永遠在 lock 外執行，與
   `setup_devices` / `get_info` 的既有鎖分層一致。

5. **Follower fail-fast**。同一 identity 已被任何 close API claim 時，第二個呼叫者
   立即收到 `DeviceCloseInProgressError(names)`，不等待、不重複 close。

6. **Failure 聚合與 BaseException cleanup**。ordinary failure 包裝為
   `DeviceCloseFailure(names, cause)`（names 含該 identity 的所有 aliases），
   release claim 且保留 entries 供重試；`BaseException` 不包裝、直接傳播，但一樣
   release claim 並完成 cleanup。`close_all_devices()` snapshot 後 identity 去重，
   嘗試 close 所有可 claim 的 identities；ordinary failures 與 in-progress
   identities 不阻斷其餘 entries，全部嘗試完後以 built-in `ExceptionGroup` 聚合
   named failures 與 in-progress identities 拋出；empty registry 是 no-op，
   snapshot 之後新增的 device 不在 batch。

7. **Notebook lifecycle：device disconnect 成功後才 close ResourceManager**。
   `notebook_md/single_qubit.md` 的 re-init cell 先
   `GlobalDeviceManager.close_all_devices()`，成功後才 `resource_manager.close()`，
   之後才建立新的 `pyvisa.ResourceManager()`；recreation cells 在 construct /
   register 前 `close_device(name, ignore_missing=True)`，失敗即 abort；final
   disconnect cell 依序 close devices → close RM → `resource_manager = None`。
   device close 失敗就不 close RM；RM close 失敗就保留 handle。disconnect 流程不
   經 output_off / ramp / set value / finally-through-device-locks。

## Consequences

- 單一 close 路徑，無散落 caller；並發 close 不會雙重執行或破壞 registry 一致性。
- Ordinary failure 保留 entries，重試安全；in-flight 競爭以 fail-fast error 明示。
- 長時間 close I/O 不阻塞其他 registry lookup，thread-safety 分層與既有
  `setup_devices` / `get_info` 一致。
- Notebook 重跑 / 斷線不殘留 session、stale aliases 或未關閉的 ResourceManager；
  session 生命周期與 hardware state 政策互不牽動。

## Rejected alternatives

- **用 `ResourceManager.close()` 取代 per-device close**：registry 會殘留 stale
  objects，也繞過 device lock semantics。
- **disconnect 時自動 RF off 或 reset value**：connection lifecycle 不是
  hardware-state policy。
- **在 registry lock 內執行 close**：長時間 I/O 阻塞其他 lookup，且可能跨鎖等待。
