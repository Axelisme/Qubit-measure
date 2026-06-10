# Findings：Python 3.9 → 3.13 升級

**最後更新：** 2026-06-11（兩輪 Explore 偵查，commit a46fb63b）

## Labber 依賴現狀（Explore 報告 1）

- **全 repo（lib/script/tests/notebooks）無任何 `import Labber` / `from Labber` / `Labber.` 使用點**。grep 僅命中 `labber_io.py` docstring 的文字提及。
- Labber 殘留位置只有宣告層：
  - `pyproject.toml:38` — client extra `"labber @ git+https://github.com/Axelisme/labber_api.git"`
  - `uv.lock:1568-1570, 4240, 4283, 4334, 4377` — 對應 lock entries（`uv lock` 會自動清）
- `datasaver.py` 已全面走 `labber_io`：
  - `datasaver.py:124` `from .labber_io import save_labber_data`（`save_local_data` 內延遲 import）
  - `datasaver.py:163` `from .labber_io import load_labber_data`（`load_local_data` 內）
  - `datasaver.py:190` **bug**：`from labber_io import load_labber_data`（bare import，`load_comment` 內；sys.path 沒有該模組時會炸）
- `labber_io.py` 的 API（`save_labber_data`/`save_labber_trace_data`/`load_labber_data`/`LabberData`）完整覆蓋 repo 所有存讀檔需求，無缺口。
- **tests/ 下沒有任何 labber_io 測試**；datasaver 經 ~65 個檔案間接依賴（experiment/v2 幾乎全部、三個 GUI 的 load/save service、tests conftest fixtures）。
- labber 套件本身依賴 PyQt5/attrdict/future——移除後這些 transitive deps 一併消失。

## 版本鎖定點與依賴阻礙（Explore 報告 2）

### 版本宣告點

- `.python-version:1` = `3.9`（唯一硬鎖）
- `pyproject.toml:6` `requires-python = ">=3.9"`（下界，不擋升級）
- `uv.lock` 已含 3.13/3.14 resolution markers（uv 多版本 resolve，不需重 bootstrap）
- 無 `.github/workflows/`（無 CI）、無 Dockerfile；ruff 無 `target-version`、pyright 無 `pythonVersion`（皆自動跟 interpreter / requires-python）
- README 未宣告版本；CLAUDE.md 寫「Python 版本鎖定 3.9」需在收尾更新

### Hard blockers

| 依賴 | 位置 | 問題 | 解法 |
|------|------|------|------|
| `numpy==1.23.5` | pyproject:18(server),39(client) | wheel 最高 cp311，3.12+ source build 必敗（舊 Cython ABI） | client 改 `>=1.26`；server extra 保留（板端用） |
| `matplotlib==3.9` | pyproject:40 | wheel 最高 cp312 | 升 3.10+（有 cp313 wheel） |
| `qiskit_metal`（design extra） | pyproject | 依賴 PySide2 5.15.2.1，wheel 只到 cp310-abi3；upstream 2022 停更 | D1：建議整個 extra 移除 |

### 需驗證項

- **Pyro4 4.82**（client extra；`lib/zcu_tools/remote/pyro.py`、`script/start_server.py` 使用）：純 Python wheel 可裝，但其 `Pyro4/utils/httpgateway.py:28` `import cgi`（3.13 已移除）。本 repo 只用 Daemon/Proxy/naming 主路徑、不碰 httpgateway；需驗證 `import Pyro4` 不會 eager-load 它。Pyro4 已停維護，Pyro5 遷移列為獨立工作（D3）。
- **qick @ git**（鎖 commit 313c74af）：本身不 pin Python，但需確認對 numpy>=1.26（或 2.x）相容。
- **labber_api 的 PyQt5 transitive dep**：移除 labber 後不再相關。

### 不受影響

- scikit-optimize 0.10.2（純 Python；`auto_optimize.py`、`jpa_optimizer.py` 用 `skopt`）、numba 0.65.1、scipy 1.15.3、scqubits、qutip、kaleido 1.3.0、pydantic 2、flask、h5py、PyQt6/qtpy——皆有 cp313 支援。
- pytest 9.0.3 工具鏈；231 個測試檔無結構性障礙。

### Server vs client 架構

- ZCU216 板上跑 PYNQ（自帶 Python 3.8/3.10 依 image），`server` extra 是板端 constraint；`remote/pyro.py:56-82` `start_server()` 在板上執行。升級只動 client；Pyro4 wire protocol 跨版本互通（serpent serializer），client 3.13 ↔ board 舊 Python 理論上不受影響，但列入 Phase 3 驗證。

### 3.9 時代語法痕跡

- 無 `sys.version_info` 檢查；無使用 3.13 已移除 stdlib（distutils/imp/asyncore/telnetlib/cgi）——只有 Pyro4 第三方內部有 cgi。
- `from __future__ import annotations`：686 處，3.13 下行為不變，非 blocker（技術債記錄）。
- 舊式 `typing.List/Dict/Optional/Union` import：~149 行；runtime `typing.Union` + `get_origin` 判斷在 `gui/app/main/services/remote/dispatch.py:826`、`gui/app/main/adapter/analyze_params.py:46`（D4 若做現代化需人工複核這兩處）。

## Phase 1 發現

- **labber_io 既有 bug**（測試定錨時發現）：`_read_trace_log` 對「無外軸、等長、Nentries>1」的 trace 檔 reshape 邏輯有誤會 raise ValueError；已以 `xfail(strict=True)` 鎖在 `tests/utils/test_labber_io.py`，待後續修復（屬 labber_io 本身，與升級無關，可在 Phase 4 順手修或另案）。
- 移除 labber 宣告連帶清掉 attrdict/future/msgpack/PyQt5(5.15) 等 transitive deps。

## Phase 2 發現

- numpy 1.23→2.0.2 在 **runtime 層零破壞**（repo 沒用任何被移除的 np 別名，全套件測試直接綠），破壞全在 **typing 層**：numpy 2 stubs 變嚴格掀出 83 個 pyright 錯（`get_xdata()` 回傳 ArrayLike 不可 index、`np.floating` 不再可直接餵 mpl 的 float 參數、builtin `divmod` 無 ndarray overload 等）。已全修。
- pandas 需配套升（2.1.0 wheel 是 numpy 1.x ABI 編譯，import 即炸）→ 2.3.3。
- **agent 驗證方法教訓**：用 git stash 驗「錯誤是否既有」對依賴升級無效（stash 不還原 venv 套件）；正確做法是直接比對升級前基準數字（Phase 1 收尾 pyright=0）。

## Phase 3/4 發現

- **PyQt6 6.11「回歸」是誤判**（investigator 用 6.10.2 對照實驗排除）：兩個 GUI 測試的 Aborted 真因是 **PyQt 對「C++ 呼入的 slot/virtual 內未捕捉 Python 例外」一律 qFatal()/abort**（v5.5 起一貫行為），而例外來自兩個早於升級的測試 fixture 契約瑕疵：
  - `test_device_dialog.py`：`info_mock=MagicMock()` 冒充 `FakeDeviceInfo` → `device_dialog.py:95` isinstance assert 在 slot 內炸。
  - `test_main_window_ui.py`：mock 了已不存在的 `ctrl.get_persisted_left_panel_width`（真實 API 自 c766ae0e 起是 `get_persisted_startup().left_panel_width`）→ `min(MagicMock,int)` TypeError 在 showEvent 內炸。
  - 處置：**不 pin PyQt6**，修測試 fixture。診斷技巧：abort 的真 traceback 要 faulthandler standalone 重現才看得到（pytest 輸出被截）。
- `tests/gui/services/test_background.py` teardown 崩潰：測試建 `BackgroundService` 未按 conftest 約定 quiesce，3.13 GC 更積極時 queued signal use-after-free。
- 3.13 切換掀出 pyright 36 errors（runner 的 `A | B` alias 當 type[T] 傳、EllipsisType sentinel、numpy 2.4 收窄、qutip overload）；agent 兩度聲稱「既有」皆不實（基準明確為 0）。
- qick parser.py 在 3.13 下大量 SyntaxWarning（上游 regex 未加 r""），無功能影響。

## 決策記錄

- （2026-06-11）計劃建立。
- （2026-06-11）用戶定錨 D1–D6（見 task_plan.md）：design extra 移除、`>=3.13`、Pyro4 只驗證、typing 留 Phase 5、numpy/matplotlib 無 pin 升最新、**server extra 整個移除**（板端 Python 3.8 用 sys.path 注入 zcu_tools + script/start_server.py + bitfiles，不經 pip；start_server.py 程式碼需維持 3.8 相容）。
- （2026-06-11）Branch 策略：全在 `py13` branch 做，端到端測過再 merge main；可動所有 tracked 檔案。
- 階段順序刻意「先測試定錨、再解 pin（仍在 3.9）、最後切直譯器」——分離變因：任何 pytest 紅燈都能歸因到單一變更（pin 解鎖 vs 直譯器切換）。
