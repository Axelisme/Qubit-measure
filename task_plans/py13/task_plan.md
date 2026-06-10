# 任務計畫：Python 3.9 → 3.13 升級

**最後更新：** 2026-06-11（計劃建立，現狀偵查完成，待用戶定奪 D1–D4 後進入實作）

## 目標

將本 repo 的 Python 版本從 3.9 升級到 3.13。

**升級成立的前提（用戶已給）**：原先鎖 3.9 的唯一原因是存讀檔依賴外部套件 Labber（最高支援 3.9）；現已有等效自寫工具 `lib/zcu_tools/utils/labber_io.py`（只依賴 h5py+numpy，支援 3.9–3.14+，與 Labber 1.8.6 byte-compatible），因此可移除 Labber 依賴並升級整個 repo。

**範圍界定**：升級對象是 **client 端（工作站）**。ZCU216 板上 server 跑 PYNQ 自帶的 Python 環境，不受本 repo `.python-version` 控制，`server` extra（`numpy==1.23.5`）保留給板端、不在升級範圍。

## 現狀總結（偵查結論，細節見 findings.md）

- **原始碼已零 Labber 依賴**：全 repo 無任何 `import Labber`；`datasaver.py` 已走 `labber_io`。Labber 只殘留在 `pyproject.toml:38`（client extra）與 `uv.lock`。
- **Hard blockers（裝不起來）**：`numpy==1.23.5`（無 cp312+ wheel）、`matplotlib==3.9`（無 cp313 wheel）、`design` extra（qiskit_metal→PySide2 只到 cp310）。
- **需驗證項**：Pyro4 在 3.13 的 `import` 路徑（其 httpgateway 用了已移除的 `cgi`，但本 repo 不用該路徑）；qick git 版對 numpy>=1.26/2.x 的相容性。
- **已知小 bug**：`datasaver.py:190` `load_comment` 用 bare `from labber_io import ...`（應為 relative import）。
- **labber_io 無任何測試**：它是全 repo 存讀檔的 backbone（~65 個檔案經 datasaver 間接依賴），升級前需補測試定錨行為。
- **無 CI**；ruff/pyright 無版本硬鎖，跟著 interpreter / requires-python 自動走。

## 設計準則（沿用專案 CLAUDE.md）

- Fast Fail、責任明確、最小驚訝、強型別、無 legacy/相容性邏輯。
- 不臆測架構；決策點交用戶定奪。

## 待用戶定奪（D1–D4）

- **D1 `design` extra 處置**：qiskit_metal（依賴 PySide2，cp310 為止）在 3.11+ 完全裝不起來且 upstream 停更。選項：(a) 從 pyproject 移除（含 `all` extra 的引用）；(b) 保留宣告但文件註明只能在舊環境裝。傾向 (a)（無 legacy 原則）。
- **D2 `requires-python` 目標**：`.python-version` 改 `3.13` 沒有爭議；但 `requires-python` 要改 `>=3.13`（單一支援版本，最乾淨）還是放寬如 `>=3.10`（lock 檔需多版本 resolve）？傾向 `>=3.13`。
- **D3 Pyro4 處置**：預設本次只「驗證 Pyro4 在 3.13 可用」（主路徑 Daemon/Proxy/naming 不碰 cgi）；遷移 Pyro5 是獨立工作不在本計劃。若驗證失敗再回報升級方案。
- **D4 舊式 typing 現代化**：~149 個 import 行用 `List/Dict/Optional/Union`（3.13 下合法、僅 deprecated）。選項：(a) 本次不動（最小變更）；(b) 升級完成後加一個 Phase 用 ruff `UP006/UP007` 自動修。傾向 (a) 先完成升級，(b) 另開收尾 Phase 由用戶決定。

## 階段規劃

### Phase 1：labber_io 測試定錨 + Labber 依賴移除 ⬜

升級前先把存讀檔 backbone 鎖死，確保後續 numpy/python 升級不悄悄改變行為。

1. 補 `tests/utils/test_labber_io.py`：1D/2D/3D uniform grid round-trip、ragged trace round-trip、metadata（comment/tags/project/user/timestamps）、多 log 堆疊讀取、`.h5`/`.hdf5` 副檔名處理。
2. 補/確認 `datasaver.py` 的 save/load wrapper 測試覆蓋（`save_local_data`/`load_local_data`/`load_comment`）。
3. 修 `datasaver.py:190` bare import → relative import（順手，屬 bug fix）。
4. 從 `pyproject.toml` client extra 移除 `labber @ git+...`，`uv lock` 更新。
5. 驗證：在現行 3.9 venv 下 pytest 全綠（基準線）。

### Phase 2：依賴 pin 解鎖（仍在 3.9 上做）⬜

1. client extra：`numpy==1.23.5` → `numpy>=1.26`（或視 qick 相容性定上界）；`matplotlib==3.9` → 有 cp313 wheel 的版本（3.10+）。`server` extra 的 numpy pin 不動。
2. 依 D1 處置 `design` extra。
3. 委派驗證：qick git 版對新 numpy 的相容性（讀其 setup 宣告與已知 issue）；matplotlib 3.10 對 gui/liveplot 的 API 影響面。
4. `uv lock` + 在 3.9 下 `uv sync` + pytest 全綠（確認 pin 解鎖本身不破壞 3.9 行為，分離變因）。

### Phase 3：切換直譯器到 3.13 ⬜

1. `.python-version` → `3.13`；依 D2 更新 `requires-python`；`uv lock`、重建 `.venv`、`uv sync --extra all`（或扣除 design）。
2. 驗證 Pyro4 在 3.13 import/基本 Proxy 路徑可用（D3）。
3. 全套件 import smoke（`zcu_tools` 各子套件 import 一輪），修 import-time 破壞。

### Phase 4：全量驗證與修復 ⬜

1. `pyright`（lib/tests/script）歸零 —— 3.13 的 stdlib stub 變化可能掀出新錯。
2. `pytest`（231 個測試檔）全綠；失敗逐一委派根因診斷後修復。
3. `ruff check --select I --fix && ruff format`。
4. GUI smoke：三個 GUI app（measure/fluxdep/dispersive）launch + MCP `state_check` 各跑一輪。
5. 更新 CLAUDE.md 的「Python 版本鎖定 3.9」描述、相關 `AI_NOTE.md`、本三件套收尾。

### Phase 5（optional，依 D4）：typing 現代化 ⬜

ruff `UP006/UP007` 批次修 + 人工複核 runtime `get_origin` 判斷點（`dispatch.py:826`、`analyze_params.py:46`）。

## 遇到的錯誤

| 錯誤 | 嘗試次數 | 解決方案 |
|------|---------|---------|
| （尚無） | | |
