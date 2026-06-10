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

## 決策記錄

- （2026-06-11）計劃建立；D1–D4 開放待用戶定奪，見 task_plan.md。
- 階段順序刻意「先測試定錨、再解 pin（仍在 3.9）、最後切直譯器」——分離變因：任何 pytest 紅燈都能歸因到單一變更（pin 解鎖 vs 直譯器切換）。
