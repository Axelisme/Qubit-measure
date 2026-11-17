---
date: 2026-11-15
status: active
superseded_by: null
related:
  - ../adr/0072-module-boundaries.md
  - ../adr/0080-datafile-generic-container.md
  - 2026-11-15-simulator-test-measurement-layering.md
  - 2026-11-14-lab-dependency-boundary.md
  - 2026-11-17-simulate-closure-expand-order.md
basis:
  repo_commit: c927e635e8a35a4b30081539793bcc6f2061d5a6
  historical_repo_commit: 29ffa7956ce8c5808dc1ba65bb4153d46e1ef3db
  external_sources: none
---

# 模擬器封閉成子模塊

## 結論

`simulate` 是封閉子模塊，前端與 environment 共用的邊界型別歸它。具體裝置模型由使用方經介面聲明。只收物理參數的元件留在 `simulate`，裝置怎麼組合則由使用方負責。整包獨立成 package 是尚未建的方向。決策、使用者原話、例子與適用邊界由 [ADR-0072](../adr/0072-module-boundaries.md#模擬器是封閉的子模塊模型由使用方聲明) 擁有。

## 問題

2026-11-15 討論[模擬器測試與量測分層](2026-11-15-simulator-test-measurement-layering.md)時，量測工具要量真實 FTF 模型，又只能依賴模擬器。使用者先說明 package 方向，原話見 ADR-0072。

當時的事實取自 2026-11-15 main，當日最後一個 commit 為 `29ffa7956ce8c5808dc1ba65bb4153d46e1ef3db`：

- `simulate` 往外只 import 兩處。`zcu_tools.core.simulation` 有 8 處，使用 CancellationQuery、SamplingIdentity、ChannelSignals、RoundSignals 等型別；該模組 docstring 寫「Immutable numeric acquisition boundary shared by frontend and environments」。另一處是 `_numba_cache.py` 從 `zcu_tools.utils.cache` 取 user_cache_root。
- `simulators` 往外 import core.simulation 18 處、program.v2 10 處、resources.simulation_schema 8 處、device 5 處、experiment.readout_plan 4 處。20 個檔案 import program、device、experiment 或 resources，因此它不能跟 `simulate` 一起封閉。
- `ModelSource.__init__(initial_density, build: Callable[[FluxLevel], SecularModel], expansion=None, ...)` 的 docstring 寫「build: pure callable supplied by the model package」。Evolution 層已經使用依賴注入，FTFSource、SingleFluxoniumSource 是使用方的包裝。
- 真實模型不縮小的要求，讓量測工具必須拿得到真實模型的 ModelSource。注入能做到這點，而不讓工具 import FTF。

## 考慮過的選項與比較

範圍題由使用者選 C，並自行寫出內容，原話見 ADR-0072。原始工作檔未入庫，選項 A、B 原文未保留，不補猜。

SingleFluxoniumModel 有兩個位置選項。A 留在模擬器當元件庫，B 搬到使用方。使用者答「1A」。它只收物理參數，不綁裝置佈局，ADR-0072 本來就讓 `simulate` 保留元件 Hamiltonian。

預測題的選項 A 是量真實 FTF 時由使用方把模型聲明注入通用工具，工具不 import FTF。使用者答「A」。

同日補充的固定場景環境約束，仍依 [Lab 依賴討論](2026-11-14-lab-dependency-boundary.md)與 [ADR-0072 的使用者空間依賴原則](../adr/0072-module-boundaries.md#使用者空間的依賴原則)，不在這裡重定義。

先例立場：

- 遵循 [ADR-0080](../adr/0080-datafile-generic-container.md) 的 datafile 獨立性。`simulate-closed` 使用與 `datafile-standalone` 同形的 contract。
- 遵循 ADR-0072 對 `simulate` 與 `simulators` 的定位，沒有把裝置組合搬進數值工具。
- 遵循量表工具複製 pulse 序列而不 import lab 的方向，出處是上述 2026-11-14 Lab 依賴討論。

## 被否決的選項與理由

把 SingleFluxoniumModel 搬到使用方被否決。它不帶裝置佈局或接線，沒有移出模擬器元件庫的理由。

讓通用量測工具直接 import FTF 不符合封閉方向。模型由使用方注入，工具只依賴聲明介面。`simulators` 已有 framework 組裝依賴，不能當作與 `simulate` 同一個封閉 package。

範圍題 A、B 的選項原文未保留，無法再補寫它們被否決的細節。

## 留下的問題

獨立 package 的名稱、發佈與版本、模型環境的歸屬，以及 prediction 工具是否一起拆出，仍未決定。問題清單見[模擬器獨立成 package](../adr/draft/simulator-standalone-package.md#尚未決定)。

## 依據

歷史事實取自 frontmatter 的 historical_repo_commit。原始工具分層討論筆記未入庫；本文保留了決策需要的數字、選項與回答。

## 後記

2026-11-17 升級文件時補記：task 已完成封閉實作，邊界型別原樣搬進 `simulate.boundary`，與 main 原檔 bytes 相同，`simulate-closed` contract 已在 integration 生效。本文核對的實作 commit 是 `c927e635e8a35a4b30081539793bcc6f2061d5a6`，尚未合併到 main，不把它記為 main 落地 SHA。搬遷的切票順序見[封閉 simulate 的搬遷順序](2026-11-17-simulate-closure-expand-order.md)。
