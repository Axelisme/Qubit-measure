# `tests/` — test suite

**Last updated:** 2026-10-19，跨平台 GUI cache 隔離

本頁說明 `tests/` 的案例歸屬、fixture 與搬遷方法。測試範圍和斷言的品質判準見 [程式碼品質](../docs/code-quality.md)，環境和驗證流程見 [AGENTS.md](../AGENTS.md)。後半保留 GUI／硬體測試的局部注意事項；不要把歷史案例當成新測試的範本。

## 放置新增測試

1. 先找同一個可觀察行為或契約的既有測試檔。新增 regression 優先放在該檔；同一 owner
   有不同且穩定的責任時，可以拆成另一個 `test_*.py`，不按 ticket、phase 或 part 切檔。
   `T1` 等名稱也可能是真實物理量，依情境判讀，不靠字面判為 ticket 名。
2. 目錄依 owning module，沿用 [AGENTS.md](../AGENTS.md) 的路徑對應：`tests/scripts/`、
   `tests/tools/`、`tests/mcp_servers/` 和 `tests/migrations/` 對應 repo root 同名目錄，其餘框架測試對應 `lib/zcu_tools/` 下的模組。
   `contract`、`parity` 是既有保留段，該段以下豁免，但前綴仍須對應；不因新案例建立新的豁免目錄。
   跨模組契約放在擁有整合行為的模組目錄，不按每個參與模組複製案例。
   Notebook 專用且已抽離的 helper 仍在其模組路徑驗證介面，不必提升為通用工具。
3. 案例按情境與預期結果命名，從模組接縫或公開契約檢查可觀察結果。
   文件、靜態內容、設定值及腳本旗標直接審閱；既有白箱或靜態檢查案例不因本頁宣稱已搬遷。
   不新增頂層 unit／integration／e2e 目錄，亦不測 Notebook／腳本的主要內容。

框架測試用最小合成 workflow／adapter，不 import `zcu_lab`。Adapter、workflow 與 recipes 的個別實驗流程與預設值不寫測試；只因實驗定義改動而失敗的測試應刪除，不修成固定 lab 行為。宿主最高接縫是 caller 的註冊清單，再從 Window、RPC 或 MCP 驗證結果。Review 檢查 tests 沒有 import `zcu_lab`，必要時以 grep 輔助。不暫移目錄跑 pytest，也不新增 `ZCU_LAB` 環境變數或 import hook，見 [模組定位 ADR](../docs/adr/0072-module-boundaries.md#分層)。

## Fixture 與測試支援碼

Fixture 放在能覆蓋使用者的最小合理目錄：只供一個檔案用的留在檔內，跨檔共用時
放最近的 `conftest.py`。`conftest.py` 管理 pytest 注入及生命週期；一般 builder、fake 或
recording adapter 放明確可 import 的支援模組，不把 `conftest.py` 當 library，也不從另一個
`test_*.py` 匯入。Fixture 的可見範圍取決於所在目錄，`scope=` 則決定實例的生命週期；
不要為了少建幾次物件把 fixture 提升到更廣的可見範圍。Root fixture 必須具有全套件用途。
相似 setup 只有在代表相同語意時才共用；不同責任不為減少行數而塞入帶多個旗標的萬用 fixture。

## UI 驗證

有 UI 改動的票，驗收須包含元件驅動情境。情境使用
[`ComponentDriver`](gui/_user_input.py)，依使用者可見文字或既有 objectName 找元件，
以真實鍵盤／滑鼠事件操作。Disabled、hidden 或完全被 scroll viewport 裁掉的元件不繞過。
可見性依 Qt visible region 判斷，不偵測 OS 層的其他視窗遮蔽。
啟動與 teardown 可用公開 fixture；情境本身只經 driver 操作與觀察。
直接呼叫 Controller／`MainWindow` 指令方法的 smoke 不算 UI 驗證，review 須標出這項缺口。

判定只讀 driver 的具名可見狀態快照，包括底部 status、元件 enabled／visible／tooltip、
可見對話框及其文字、欄位顯示值、圖例文字、圖上軸標與可見曲線數。
曲線觀察只讀公開 Matplotlib artist 的可見狀態，仍須看 PNG 確認外觀，不拿結果資料當判定。
不讀 Controller、state store 或私有屬性。
背景工作依下方的 [Qt owner-loop 等待](#qt-owner-loop-等待)規則，用公開終局條件等待。

框架情境用合成 adapter，放在對應 owner 的 `tests/` 路徑。
實驗定義的走查腳本用同一個 driver，放在各 task scratch，不 commit。
每一步將視窗截圖存為 PNG 到呼叫端指定的 task scratch 或測試 `tmp_path`。
Agent 在 review-done 前須跑完情境並親自看過每步截圖，留下精確重跑命令、截圖路徑與判讀結論。
使用者走查只判斷外觀與操作手感，不承擔行為測試。

## 搬遷驗收

搬檔前列出舊檔到新檔的案例映射，確認每個情境、斷言與 marker 都有對應；搬檔後
比較 pytest collection 與 marker，核對 fixture lookup、覆寫與生命週期，並跑受影響選集及共同父目錄。
涉及共享狀態時，驗證不同執行順序與平行選集。先做純搬檔，再另外審查斷言改寫與重複案例刪除。
檔案移動可能改變 `conftest.py` 的 fixture lookup；案例總數相同不能取代上述核對。
本頁不代表現有測試已搬遷。

## 執行測試套件

受管理 worktree 的 Python 指令按 [AGENTS.md](../AGENTS.md) 使用 `uv run --directory <worktree> --no-sync --`。
選集與平行化由 pytest CLI 決定。開始前確認所用環境安裝 GUI/client 測試需要的 optional dependency。

pytest 全域 warning filter 只放第三方 noise（例如 `qick` / `scqubits` 在 Python 3.13
import 時的 invalid-escape `SyntaxWarning`）。本 repo 的 production warning 不應全域 suppress；
若 integration test 允許某個 fit fallback warning，使用 test-local `filterwarnings` marker
標註該測試的預期退化。

pytest 預設排除 `slow`。ticket 內只跑受影響選集，不跑 slow，最多使用兩個 worker：

```bash
uv run --directory <worktree> --no-sync -- pytest <affected-paths> -n 2
```

task 進 land review 前才跑整套驗證。CLI 的 `-m ""` 取代預設 marker 過濾，包含 slow 與非 slow。
完整測試固定使用四個 worker。重檢查經過全機資源限制，同 task 用 lock 串行：

```bash
flock <task-scratch>/heavy-task.lock <repo>/.agent_state/tools/heavy.sh \
  uv run --directory <worktree> --no-sync -- \
  pytest -n 4 --dist=worksteal -m "" --durations=25
```

`-m slow` 只驗證 slow 選集，不是整套驗證。`requires_loopback` 等能力 marker 只宣告資源需求，
不用來跳過慢測試。slow 門檻依整套實測決定，數據與理由記在 task receipt。
fixture 過重、重複計算或不必要的 sleep／輪詢要直接修，不用 slow 隱藏成本。
只為需要保留的昂貴驗證加 slow，不做全域逐 node skip。

### 測試成本

每個測試函數的計算只用單核心。一個測試內部的 worker、thread 或 process 總數最多兩個，用於驗證多 worker 契約。每個測試在 `-n 0` 下少於 30 s。
測試不跑慢的模擬或擬合。行為改用小型合成模型（少數能階、短 timeline）、解析解、不變性或最小 case 對照 reference 來確定。
完整精度與效能量測放在 task 的量測腳本，不放進測試套件。新增測試時以 `--durations=0` 確認耗時。

`tests/conftest.py` 在每個 xdist worker 啟動時，把
`OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS` 預設設為 `"1"`，
避免 worker 與 BLAS 執行緒超額競爭。Serial 跑法不設這些變數，適合 debug。
Qt GUI 子套件可選 `tests/gui tests/gui/app/fluxdep tests/gui/app/dispersive`。

### GUI cache 隔離

`tests/conftest.py` 在第三方套件、子目錄 conftest 與產品模組 import 前設定
`ZCU_TOOLS_CACHE_DIR`，指向 worktree 內獨立的暫存目錄。
建立與退出清理由可在第三方 import 前載入的 `tests/_cache_environment.py` 負責。
GUI persistence 與 session discovery 在 Linux、Windows 和 macOS 都使用這個根目錄。
每個 xdist worker 各自建立目錄，不使用會被 pytest 清掉的 basetemp。
這些目錄保留到 process 結束，再由 `atexit` 清理。
巢狀 `pytest.main()` session 的 teardown 不還原 `ZCU_TOOLS_CACHE_DIR`，後續 cleanup 仍使用隔離目錄。

### Background work — 測試 teardown 必要模式

Fixture 必須在 `deleteLater()` 或放 GC 前等待背景工作與 queued delivery。
持有 measure Controller 時呼叫 `close()`。直接持有 `BackgroundRunner` 時呼叫
`quiesce()`。

原因：worker 的 `on_done` 是跨執行緒 queued signal；如果 runner 在 delivery 進到 main-thread queue
之後就被 GC，下一次 `processEvents()` 會把訊號分發到已釋放的 C++ 物件 → segfault。

持有 measure `Controller` 的 fixture 呼叫公開 `close()`，不讀取內部 runner。
它先永久停止 admission，再等待 worker 與 queued owner deliveries。逾時會 raise
`TimeoutError`，teardown 不應吞掉例外或當作已完成。成功後可重複呼叫。

```python
@pytest.fixture
def controller_view(qapp):
    ctrl, view = make_controller_view()
    yield ctrl, view
    ctrl.close()
    view.deleteLater()
    qapp.processEvents()
```

直接持有其他 runner 的 fixture 繼續呼叫該 runner 的公開 lifecycle API，不套用
Controller 的 terminal close。見 `tests/gui/session/services/test_device_manager.py`
的 `_quiesce_services` fixture。

### 等待、時間與完成條件

先指出需要建立的先後關係，再用協作者的抵達訊號、受控 owner queue、明確 release 與
join 建立因果。從公開接縫驗證結果，不以正向短時限抵達、sleep 後斷言或實測耗時上界
判對錯。拉長 timeout、skip 或 retry 不能取代同步。呼叫開始前 set 的 attempted Event
只表示準備呼叫，不證明已入列、取得 admission 或進入鎖競爭。

等待必須由已安排且設計上必達的事件結束，例如已提交 RPC 的 responder 抵達、release 後的
worker 完成或 owner 終局交付。先建立這項因果，再加數十秒級 watchdog 防卡死，通常設為 30 秒。
在旁邊一行說明必達原因。Watchdog 不是效能門檻，也不能把「收到 cancel 請求」當作
「工作已停止」。失敗路徑仍須 finally release 全部 blocker、join worker、傳回 worker 例外，
再關閉 session 或釋放 Qt 物件。Background teardown 繼續使用上節的公開 close／quiesce。

驗證 timeout、debounce 或時鐘契約時，控制外部 timer／虛擬時鐘並明確交付到期事件，
保留 timeout 的返回值、副作用與後續可完成行為，不等待指定的真實 duration。
只有公開契約保證可觀察同一分支時才用零時限 poll；timeout 不代表 cancel，也不證明 worker
已抵達。物理 duration、production 計時與離線 benchmark 不因名稱含時間就改寫。

替身只控制協作者的外部效果，不替換被測邏輯。對外 RPC／transport 的公開方法可以加前置
delegating wrapper，但必須同時符合下列條件：

- 以相同參數呼叫原方法，原樣 return 或 raise，不偽造回覆。
- 只在呼叫之前暫停、記錄或 set Event。
- 不讀寫私有 lock、state 或 owner 內部 helper。
- 不跳過或替換 admission、序列化、cancel 或 save 實作。

這是邊界觀察，不是任意 monkeypatch 的許可。Mathtext 鎖測試對第三方
`matplotlib._mathtext.Parser.parse` 的委派觀察只限該外部協作者邊界；它不允許讀寫其餘
私有屬性、cache 或 lock，也不適用其他第三方私有模組。該測試從公開 parse 入口走真鎖，
保留 install／prewarm、fresh expression、patch 還原與 release／join；Matplotlib 升級時須
重新核對內部路徑，屬性消失應明確失敗，不 skip 或 fallback。

### 負向敏感窗與事件終止取樣

「一段時間沒發生」只在找不到公開抵達觀察點、且只驗證某事未發生時，才可保留為短負向
敏感窗。逐處寫明公開排隊抵達點的缺口。時間只影響抓錯機率，正確實作不會因負載而失敗。
不得宣稱競爭者已排隊，也不得把正向抵達等待列入此例外。不能只改成沒有同步的 is_set。

互斥案例由 responder／recording transport 記錄進出，競爭者在 holder 持有期間進入就直接失敗。
Release 後 join，檢查完整順序與最終結果。Queued-intent admission 案例仍保留 attempted 與
短負向窗，另以確定性順序和真實終局補強；不 patch 私有 lock、lock factory 或 pre-lock helper。
Blocking waiter 案例保留公開 pending poll、timeout 不取消、release 後的 owner 終局與 join；
pending poll 不能證明下一次 blocking await 已抵達。

Finish-boundary hammer 等取樣型也只限沒有公開抵達點的負向檢查。迴圈由 leader 完成、release
等事件結束，之後 join。短窗或取樣次數只影響偵測力，不以「N 次短 wait 內必須看到 X」判成功。
外圍 30 秒只防卡死，旁註必達原因。仍驗 close-once、空 registry、before／after 等確定性終局，
不宣稱取樣能命中所有錯誤交錯。需要新的 production 同步接口時，先交由任務決策者裁定，
不以移入 backlog 代替修復。

### Qt owner-loop 等待

需要真實 Qt signal／owner delivery 的測試保留該整合，統一使用
`tests/gui/_qt_waits.py::qt_until(completed, *, label="completion")`，不再新增本地 duration helper
或另一套等待 framework。它以 0ms QTimer probe 在 owner-loop turn 取樣受控 Event 或公開終局
條件，條件成立才成功。獨立 single-shot 30 秒 timer 只作 watchdog。Probe 次數或經過時間都不是
成功條件，probe 也不證明 queue arrival。Helper 會把 predicate 例外傳回 caller，並停掉兩個 timer。
不要在 predicate 重入阻塞 RPC。非 Qt 契約可使用既有 manual owner scheduler。

Debounce 測試控制外部 QTimer 排程或虛擬時鐘，不用 `QTest.qWait`、固定 duration 的
`QEventLoop + singleShot` 或 stub suppression 解決。單次 `processEvents()` 可以 flush 已交付工作
或 layout，不能當背景工作完成證據；背景完成仍需上述因果條件與公開 teardown。

---

## 現有 owner 導覽

`tests/program/v2/` 擁有 QICK compile、IR、macro、module 與 simulator 行為；
`tests/experiment/v2/` 擁有通用 runtime 與工具，`tests/experiment/workflows/` 擁有泛型 Engine、effects 與共用匯出機制。
`tests/gui/app/workflows/` 使用合成註冊清單驗證宿主；`tests/gui/app/measure/adapter/` 擁有 adapter、黑箱 Form 輸出與 Run 框架契約，`tests/gui/form_adapter/` 擁有 core cfg 生成 Form 的契約。`tests/gui/` 與各 app GUI 目錄擁有 UI、service、remote
接縫；`tests/mcp/` 擁有 MCP bridge 與操作契約。`tests/resources/`、`tests/analysis/`、
`tests/notebook/`、`tests/datafile/` 與 `tests/utils/` 分別擁有其路徑對應模組的測試。`tests/migrations/` 擁有 root `migrations/` 的離線轉換與 legacy reader 測試。
例如 `tests/program/v2/modules/test_registry.py` 測 `PulseRegistry` 的 pulse 定義 SHA256 去重，
與同名的其他 registry 測試無關。需要定位檔案時以目前目錄及程式 owner 為準。

---

## 兩層 Fixture 策略

### Level 1 — `make_mock_soccfg()`（實作在 `lib/zcu_tools/program/v2/mocksoc.py`，由 `tests/program/v2/conftest.py` 匯入）

用真實 `QickConfig` 從純 dict 建構，不需要任何硬體。  
所有 `ModularProgramV2.__init__` → `compile()` → `_initialize/_body` 都跑真實 QICK ASM。

**Gen 規格**（`axis_signal_gen_v6`，HAS_MIXER=False）：

- `fs=12288.0 MHz`（245.76 × 50），`f_dds=12288.0`，`b_dds=32`，`b_phase=32`
- `fs_mult=50`，`fs_div=1`，`fdds_div=1`，`samps_per_clk=16`
- `has_mixer=False`，`maxv=32766`，`maxv_scale=1.0`
- nqz=1 有效頻率範圍：0–6144 MHz（f_dds 拉高到 12288 給 6 GHz 級 readout 充足 headroom，免折疊）

**Readout 規格**（`axis_readout_v2`）：

- `fs=2457.6 MHz`（245.76 × 10），`f_dds=2457.6`，`b_dds=32`
- `fs_mult=10`，`fs_div=1`，`fdds_div=1`，`decimation=1`
- `f_output=307.2 MHz`（fs / (decimation × DOWNSAMPLING=8)）
- 有效頻率範圍：0–1228.8 MHz

**為什麼不用 `axis_sg_int4_v1`？**  
這個 gen 是 `HAS_MIXER=True`，`declare_gen()` 會強制要求 `mixer_freq`，  
但 `Pulse.init_pulse()` 只在 `cfg.mixer_freq is not None` 時才傳 `mixer_freq`，  
導致 `RuntimeError: generator N has a digital mixer, but no mixer_freq was defined`。

**哪些欄位必須手動補齊？**  
`QickConfig.__init__` 不計算任何欄位，欄位由 FPGA firmware driver 設定，  
所以 mock dict 必須包含：`has_mixer`、`maxv`、`maxv_scale`、`b_phase`、  
`fs_mult`、`fs_div`、`fdds_div`、`interpolation`（gen）；  
`fs_mult`、`fs_div`、`fdds_div`、`decimation`、`has_outsel`（readout）。  
缺少任何一個就會在 `declare_gen` / `add_readoutconfig` / `freq2reg` 時 `KeyError`。

### Level 2 — `MagicMock` (`tests/program/v2/modules/conftest.py`)

用於 `modules/` 下的**純單元測試**，只測試 cfg parsing / set_param / allow_rerun 等邏輯，  
不觸發真實 QICK 編譯。`mock_prog.soccfg` 是一個最小 dict，不需完整硬體欄位。

### IR 結構測試（`tests/program/v2/ir/`）

`BasicBlockNode` / `BlockNode` / `IRLoop` 可建立不依賴 MockSoc 的輸入。
經由負責該行為的 IR pipeline 接縫檢查輸出結構或編譯結果；不要以私有
`_optimize_tree()` 的回傳物件身分判斷是否發生變更，因為它可能原地修改 `BlockNode`。

### SNR scorer helpers

`tests/experiment/v2/utils/test_snr.py` 用解析的 g/e `mean` / `covariance` / `third_moment` fixture 測 `snr_as_signal` 與 `skew_penalty`，不靠隨機 samples 或 Monte Carlo 閾值；這能直接驗證 pooled-sigma SNR、shape mismatch penalty 與 one-sided skew penalty 的公式語意。

### Experiment v2 Schedule runtime tests

`tests/experiment/v2/runtime/test_program_builder.py` 覆蓋 ProgramBuilder 建構、cfg 投影與 program-side sweep；`test_program_acquire.py` 覆蓋 integrated／decimated acquire、caller-owned program cache、progress 與 raw conversion。`test_program_retry.py` 覆蓋 retry、stop checker、failed attempt 後 stop 不再 retry 與 `ScheduleOutcome`。
`test_schedule.py` 覆蓋 typed env、host scan／repeat／batch；`test_schedule_buffers.py` 覆蓋 root buffer 註冊、child buffer 同步與 shape validation。`test_schedule_lifecycle.py` 擁有 context 與 stop token 生命週期，`test_signal_buffer_flush.py` 擁有 SignalBuffer 寫入通知、manual update 與 flush。
`test_result_tree.py` 覆蓋 executor-owned ResultTree 的 node set、direct node env event / missing-env fast-fail、child buffer、per-measurement subscription、root broadcast、flush 與 ordinary SignalBuffer regression。個別 experiment module 更接近資料編排，不新增 migration-specific tests；若要測 QICK compile 行為，放到 `tests/program/v2/` 或既有 sim integration 測試。

### Experiment device contracts

`tests/experiment/utils/test_device.py` 在選定 driver 的公開接縫驗證 flux 單位、模式、安全限制、取消與錯誤傳播。Cfg assembly tests 驗證角色到名稱的 mapping，且不讀硬體。用到 device 的合成 cfg 在自身宣告該欄位，不依賴 framework base 的 default。

GUI Guard tests 驗證 cached settings 缺失或無效時仍可 admission；Guard 沒有可觀測的 driver 接縫，不在這些案例加入硬體讀取計數。Run 與 Notebook owner tests 驗證 acquisition 前讀全部 connected drivers 一次、讀取失敗拒 Run，以及固定 metadata 經 Save 後的讀回。DeviceInfo label 逆向投影與實驗自動 setup 的專用案例已退休，手動設定與 manager 的有效 owner tests 保留。

### Device manager tests

`tests/device/test_manager_lock.py` 覆蓋 instance `DeviceManager` registry lock 只保護
registry dict、`get_info` 不被其它 device 的 setup ramp 阻塞、整批 name validation
fast-fail、以及 `setup_devices(..., cancel_signal=...)` 的顯式協作取消語意。取消測試使用真 `FakeDevice`，不註冊 MagicMock driver。

### Offline acquisition transports

Compile-only cases use `make_mock_soccfg(n_gens, n_readouts)`. Simulation acquisition
cases explicitly build `white_noise(...)` or a numeric `SimulationBinding` and use
public program acquire/get_raw. Normal unbound pipeline tests use the deterministic
transport in `tests/program/_acquisition_soc.py`. No fixture silently installs noise.
Synthetic registered experiments exercise Window/RPC Run, Stop, failure and native Save
in `tests/gui/app/measure/test_simulation_run.py`. All offline caches remain isolated.

### FTF numeric and acquisition tests

`tests/resources/test_ftf.py` verifies named YAML validation and dotted field
errors. `tests/simulate/test_ftf_model.py`, `test_joint_baths.py` and
`test_source_lines.py` own fixed-coordinate circuits, LC/Boltzmann/Gibbs references,
CP dephasing and causal source tails. Independent evolution/ADC behavior stays in
`tests/simulate/evolution/` and `test_adc.py`; model-event and multi-window
sampling tests live in the corresponding numeric owner directories.
`tests/experiment/test_ftf.py` covers explicit unit conversion, common carrier phase
and public acquire/decimated composition. `test_ftf_templates.py` covers DC readout
and the template engine contract.
The former TLS engine/frontend and tests serving only that implementation are
retired. Framework Window/RPC tests continue to use synthetic registered
experiments, not lab definitions or model-truth calibration oracles.

### Form 與持久 Spec 的測試 owner

`tests/gui/form/` 驗證共用 input、單層 describe、expression、options、defaults、state 與 preview。
`tests/gui/form_adapter/` 驗證 core cfg 生成、持久 Spec 投影、child key 保留與屏蔽欄位政策。
`tests/gui/widgets/form/` 從 FormEditor 觀察提交、解析錯誤、唯讀、群組與選項，也涵蓋共用 value-widget helpers。

Library entry 與範本的公開行為由 `tests/gui/session/test_library_entry.py` 驗證，
Inspect 與 Aliases 由 `tests/gui/session/ui/` 驗證。Measure 的 tab Form resource、
固定 capture 與 Run 在 `tests/gui/app/measure/services/`；writeback draft、Apply 與 widget
在 `tests/gui/app/measure/writeback/`；
Library／writeback 的 wire 行為由 `tests/gui/app/measure/remote/` 驗證。
Writeback 案例區分 Form batch、draft-local edits 與 target 提交，不把 selected Apply 當成跨檔原子交易。

Import ownership、來源依賴、靜態宣告與文件內容直接審閱，不以 AST、
absence assertions 或獨立程序匯入結果作測試。框架案例使用 synthetic adapter／Spec，
不 import zcu_lab，也不固定個別實驗的預設值。

### Adapter 與共用工具 tests

`tests/gui/app/measure/adapter/test_form_run.py` 驗證 Form 輸出原樣交給 adapter；
`services/test_tab_form_resource.py`、`test_tab_form_notifications.py` 與
`test_tab_cfg_application.py` 擁有固定 capture、Reset、busy／lifetime 與通知。
`tests/gui/form_adapter/` 擁有推導與持久範本契約。舊 builder、Seed wrapper 與角色表的專用測試退休。
Singleshot Rabi 數值與畫圖分別由 `tests/analysis/fitting/test_singleshot_rabi.py`
及 `tests/plotting/test_singleshot_rabi.py` 驗證。Notebook flux picker 使用 caller 提供的
`FluxPickInputs`，由 `tests/notebook/test_flux_dep.py` 驗證。

`tests/gui/app/measure/adapter/` 保留 BaseAdapter 的 capability、canonical load、
post defaults、lowering 與 writeback framework 契約。
`tests/notebook/test_adapter.py` 與 `test_plotting.py` 使用共用 recording core，驗證
source retention、nullable options、失敗隔離與 widget lifecycle，不使用具體實驗數值。

### GUI remote/control tests

`tests/gui/remote/test_lazy_broadcast.py`以barrier鎖定two-phase recipient
selection：零matching recipient不build、多client只build一次、unsubscribe/disconnect送前重驗、
late subscribe不補收舊event，以及slow-client drop budget不阻塞healthy client或改變per-client order。
remote EventBus focused tests另鎖serializer/current-path/encode lazy cost與failure logging；
diagnostic測試確認fault channel仍不受subscription gate影響。

`tests/gui/app/measure/ui/test_main_window_events.py`鎖定closed tab facts到Qt reaction的
完整call sequence、zero-reaction local edits與lazy單次snapshot；service與真實UI測試覆蓋
run去重、analysis start-rejected/failure/cancel retained-figure restore、load stale-canvas clear、
same-class form hydrate/cache，以及ModuleLibrary變更透過attached cfg draft更新run gate。

`tests/gui/test_expected_error.py`鎖定closed category、legacy RuntimeError/ValueError ancestry與
explicit concrete opt-in/exclusion；`tests/gui/app/measure/remote/test_expected_error_wire_compat.py`
以exact `(code, message, reason, data)` tuple鎖定既有handler projection，並證
`ResultScopeError`分類不依賴reason prefix。

`test_expected_error_dispatch.py`走實際shared dispatch reply path，鎖定main/off-main category
mapping、direct structured `RemoteError` passthrough、generic `data=None`與translator failure
containment。`test_unexpected_handler_errors.py`以既有handler樣本確認ordinary RuntimeError、ProviderError、I/O與
invariant failure不被降級；unexpected dispatch測試另確認controller error log保留traceback。
新增 request handler 或修改分類時，直接審閱 catch 範圍與 method entry 的靜態宣告，
再用公開 request／reply 測試確認對外錯誤碼。

`tests/mcp/measure/`擁有measure MCP tool assembly、guard、operation、timeout、bundle、
view product及lifecycle／stdio行為。每個fixture建立自己的session／bridge／tool table，
透過recording Transport觀察RPC，不patch server globals或私有helpers。
`tests/gui/app/measure/remote/test_remote_mcp_toolchain.py`保留GUI project/device/save／guide
handler契約；同目錄的事件整合測試保留真socket，驗證EventBus→bridge→session的origin。
`tests/gui/app/measure/_app_launch.py`啟動真實 `MeasureGuiBehavior`（window、remote adapter、MCP client），`tests/gui/app/measure/test_setup_launch_flow.py`與`ui/test_named_dialog_screenshot.py`用它驗證 Setup 從還原、`after_show`、MCP screenshot、工具列聚焦到關閉重開，以及 project tool 變更後的預填。
Shared exposure policy 的可觀察行為屬於 `tests/gui/remote/`。Schema 文字、tool inventory 與
script flags 用直接 review，不納入 pytest。

`tests/gui/_control_fakes.py` 提供 control facet tests 的 typed recording fakes。新增 `test_*_control.py` contract 時，偏好 recording fake + 表驅動 public forwarding contract；只在需要 Qt signal / event bus behavior 時測 event disposer、signal rebind 或 state transition，不把 `MagicMock.assert_called_once_with` 當成主要測試內容。

GUI widget tests use `tests/gui/_dialog_fakes.py::RecordingDialogPresenter` for
information messages, warnings, critical errors, confirmations, and destructive
confirmations exposed through the shared `DialogPresenter` port. Prefer
injecting this adapter into the widget under test over monkeypatching
`QMessageBox`; keep QFileDialog, QInputDialog, QMenu, or method monkeypatches
only when the tested object has no dialog-presenter boundary for that
interaction.

### GUI device service tests

`DeviceManager` 是 instance registry，入口只接受 `BaseDevice` instance。GUI service unit tests 若用
`MagicMock` driver 驗證 call interaction，注入 `tests/gui/session/services/_device_fakes.py::FakeDeviceRegistry`。
Registry CRUD 使用真 `FakeDevice`。`services/test_device_registration.py` 經真 manager 與 session owner 驗證 runtime 註冊、失敗保留與逐台 close；`app/measure/test_device_lifecycle.py` 驗證 drain 保留資源，以及 app 關閉隔離 driver 錯誤並卸載 SoC。不靠 SoC change 通知補建資源。

`DeviceService.poll_device_info(name)` 測試應視為 best-effort off-main live-read contract：memory-only、
connect/disconnect 等非 setup mutation 會 skip；`SETTING_UP` 的 selected device 可 poll current driver
info 來刷新 cache/UI，但 late delivery 若已進入非 setup mutation 不可 bump/emit。

等待 async device connect 時同時觀察 success 與 `operation_failed`，以共用 `qt_until` 等到其中
一項已交付，再斷言結果。不要用只聽 success 的裸 `QEventLoop.exec()` 或 elapsed deadline 的
`processEvents()` loop；失敗必須回到 caller，不讓 pytest 掛住。

### Golden / characterization tests

大型 golden/snapshot equality 可以保留作 characterization，但旁邊應有 focused semantic invariant 說明 load-bearing contract，例如 role set ownership、live `EvalValue` link、或 schema 宣告的是 user knobs 而非 derived cfg fields。變更 golden payload 時，先用 invariant 說明語意，再更新 payload。

測試不重寫 production default/value table。若 characterization 必須比對 default 或派生值，expected 從 production schema、constant 或 helper 取得；測試本身只寫語意 invariant 或明確測試輸入。

---

## 整合測試注意事項

### 頻率值

測試檔案 `test_modules_integration.py` 使用：

```python
GEN_FREQ = 1000.0   # MHz，nqz=1，axis_signal_gen_v6 有效
RO_FREQ  = 100.0    # MHz，axis_readout_v2 有效
```

### PulseReadout / gen_ch

`PulseReadoutCfg` 使用平鋪的 `gen_ch`、`ro_ch` 與單一 `freq`。Runtime 先宣告 generator 與 ADC，再建立 readout configuration；caller 不另建 Pulse 或 DirectReadout 配置。

### Register-driven Repeat

```python
r = Repeat("r_cnt", "n_count")   # "r_cnt" = counter reg，"n_count" = count reg（sweep 建立）
r.add_content(...)
_make_prog(modules=[r], sweep=[("n_count", 4)])
```

`Repeat(name, n_reg)` 中的 `name` 是 `OpenLoopReg.preprocess` 新建的 counter register，  
`n_reg` 是已存在的 count register（由 sweep 的 `add_loop` 建立）。  
**不能**用同一個名字 `Repeat("x", "x") + sweep=[("x", N)]`——sweep 和 OpenLoopReg 都會試著建 `"x"`，造成 `NameError: register name 'x' already exists`。

### Branch

需搭配外部 sweep loop，且 `compare_reg` 名稱要與 sweep loop 名稱一致：

```python
Branch("sel", [...], [...])  # compare_reg 預設為 "sel"
_make_prog(modules=[b], sweep=[("sel", 2)])
```

### Join / merge_max_length UserWarning

`modules/util.py::merge_max_length()` 在兩個 length 值相等且無法 reduce 時發出 UserWarning（「Detected multiple overlapping lengths」）。在測試 `Join` 時，若兩個 branch 的 `SoftDelay` delay 值相同（例如都是 `0.0`），round 後都為 `0.0`，會觸發此 warning。解法：使用不同的 delay 值（例如 `0.1` vs `0.5`），讓 `merge_max_length` 可選出明確最大值。

### LoadValue 壓縮閾值

`auto_compress` 只在 `len(values) >= 30` 時啟動。  
測試中用 64 個元素驗證壓縮路徑，3 個元素驗證非壓縮路徑。

### IR pass 測試

- `tests/program/v2/ir/test_ir_builder.py` 覆蓋 `__META__` parsing、branch case identity、loop section parsing，以及 jump label reference 不被誤判成 label definition。
- `tests/program/v2/ir/test_ir_passes_optimization.py` 等 `ir/test_ir_passes_*.py` 涵蓋結構與 label validation、label DCE、branch case normalize、constant loop unroll、hoist/peephole/timing pass。新增案例從 pipeline 結果驗證 IR 行為。
- `PeepholePass` 應保留 `NOP`；驗證輸出時分辨 `NOP` 與可清除的 `IR_` internal annotations。
- `ConstantLoopUnrollPass` 測試使用顯式 `IRLoop.trip_count`，不要用 QICK 實際 loop asm shape 當作 unroll 偵測依據。

### estimate_* 函式測試注意事項

`estimate_body_scheduled_ticks` / `estimate_flat_size` / `estimate_body_cost` 的簽名接受 `list[IRNode]`，但 Python list 是 invariant：不能把 `list[BasicBlockNode]` 直接傳入，需要明確宣告型別 `nodes: list[IRNode] = [bb1, bb2]`。

### IR block 不變量

`BasicBlockNode.__post_init__` 拒絕在 `.insts` 中放 `JumpInst`。新增 pass 測試用有效的 block
經 pipeline 檢查可觀察結果；不為 unreachable 的非法 block 繞過建構器測私有方法。

### UnrollLoopPass — jump table 觸發條件

Register-driven loop（`n=Register`）+ `available_regs` 非空 + `k_final >= 2` + `body_size > 0` 才觸發 jump table。測試中：body 用 NopInst（size=1），`pmem_budget=1024`（k_budget=1024），`cost_default=1, cost_jump_flush=40`，body_cost=1，scheduled_ticks=0，slack=-1 < 0 → k_timing=max_unroll_factor=32；k_final=min(32,1024)=32 → 觸發。

### SimplifyDispatchPass — 回傳 BlockNode 而非 BasicBlockNode

`SimplifyDispatchPass` 對 `IRDispatch(k==2)` 回傳 `BlockNode([cond_bb, fallthrough_bb])`，兩個分支均顯式 jump，不依賴 fall-through：

- `cond_bb`：`JUMP target_labels[1] -if(NZ) -op(value_reg - #0)`（小 PMEM）；big-PMEM 用 `REG_WR s15 label + JUMP [s15]`
- `fallthrough_bb`：`JUMP target_labels[0]`（小 PMEM，`BranchEliminationPass` 可消除）；big-PMEM 永遠保留

經 pipeline 驗證時檢查兩個 branch 的目標，而不是依賴舊的 `BasicBlockNode` 回傳型別。

### DmemDispatchPass — 小/大 pmem 路徑

`DmemDispatchPass` 的 guard block 依 `needs_big_jump(pmem_capacity)` 分叉：
- `pmem_capacity <= 2048`：guard 是 label-mode `JumpInst`（`label=LabelRef(last)`, `addr=None`）
- `pmem_capacity > 2048`：guard 是 `RegWriteInst(dst=s15, src=LABEL)` + `JumpInst(addr=s15)`

驗證小 pmem 路徑時提供 `PipeLineContext(config=PipeLineConfig(pmem_capacity=512))`，經 IR pipeline 觀察生成結果。

### DeadWriteEliminationPass / ZeroDelayDCEPass — disable_opt guard

`block.disable_opt=True` 應保留未優化的行為。以有效 `BasicBlockNode(..., disable_opt=True)`
作 pipeline 輸入，比較產生的 IR，而非直接斷言 `_process_block` 的呼叫或回傳。

---

## 新增測試的快速指南

### GUI analyze params 測試

`tests/gui/app/measure/adapter/test_analyze_params.py` 覆蓋 dataclass-based analyze params helper；`tests/gui/app/measure/ui/test_analyze_form.py` 覆蓋 `AnalyzeFormWidget` 的 dataclass round-trip、hydrate 不 emit、使用者編輯 emit instance。新增 GUI adapter 測試時，analysis 參數應直接使用 adapter 回傳的 params dataclass instance，不要組 raw dict 或假設 `get_analyze_params()` 可迭代。

### measure-gui canonical result load 測試

load-result feature 的 targeted tests 分散在對應 ownership：
`tests/gui/app/measure/adapter/test_base_load.py` 鎖 adapter default load contract；
`tests/gui/app/measure/services/test_load.py` 鎖 state invalidation / version bump；
`tests/gui/app/measure/ui/test_main_window_ui.py` 鎖 `Load Data...` button gate 與 file dialog；
`tests/gui/app/measure/remote/` 鎖 `tab.load_data` dispatch、tool generation 與 MCP guard deps。
`tests/mcp/measure/`覆蓋operation handle與RPC timeout policy：bounded
GUI handler timeout應回傳狀態，transport timeout應被視為連線異常。

Program v2 的 compile regression 先找 `test_modules_integration.py` 中同一契約，
以 `_make_prog(modules=[...])` 建構程式，檢查編譯結果及該情境需區分的語意。
`tests/program/v2/modules/` 的 cfg parsing 或 set_param 案例可用 `mock_prog` 隔離硬體；
從 module 接縫驗證結果，只有呼叫順序本身屬於契約時才斷言順序。

**新增 soccfg 欄位**：若 QICK 版本升級導致新的 `KeyError`，在  
`lib/zcu_tools/program/v2/mocksoc.py` 的 `_build_mock_cfg()`（內含 `_gen()` / `_readout()`）補入對應欄位即可。

---

## Cfg 解析測試的 import 注意事項

`ModuleCfgFactory`／`WaveformCfgFactory` 由 `cfg_model` 擁有，按 Spec 與成品 cfg 的 union 驗證並遞迴展開引用。測試從 schema owner 的 package 入口 import，避免直接碰 leaf base class：

```python
from zcu_tools.cfg_model import ModuleCfgFactory   # 使用 schema owner 的 package 入口
# 不要從 program runtime 或 schema leaf 匯入 factory

cfg = ModuleCfgFactory.from_raw({"type": "readout/direct", "ro_ch": 0, ...})
```

呼叫 `ModuleCfgFactory.from_raw(raw, context=...)` 進行解析。`CfgValidationContext` 明示固定的 module resolver 與 asset capability；dependency-free inline cfg 可省略 context。

**測試一個非 `Literal` 的 cfg（負面測試）**：若要在測試裡定義 `class UnknownCfg(AbsReadoutCfg): type: str = "unknown"` 來驗證 factory 錯誤路徑，重點是它不在 `ModuleCfg` Union 裡，因此 `ModuleCfgFactory.from_raw()` 不會 dispatch 到它。
