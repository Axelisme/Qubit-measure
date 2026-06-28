# liveplot 模組重點筆記

**Last updated:** 2026-06-29

Jupyter 中即時更新的 matplotlib 繪圖工具，在資料擷取過程中邊跑邊畫。

## 目前模組分層

1. `base.py` / `multi.py` — 介面與多圖組合
   - `AbsLivePlot`：所有 live plotter 的抽象基底（`clear / update / refresh` + context manager）。
   - `DummyPlot`：no-op null object。
   - `MultiLivePlot[PlotKey_T]`：組合多個 plotter，用 key 分發 `update(plot_args={key: (args...)})`；`refresh()` 呼叫 `refresh_figure(self.fig)`，由 active backend 決定具體重繪行為（JupyterBackend 用 `canvas.draw()`，FallbackBackend 用 `draw_idle`）。

2. `backend/` — backend 契約、註冊與內建 backend
   - `backend/base.py`：`LivePlotBackend` ABC，四個 abstractmethod（`make_plot_frame` / `instant_plot` / `refresh_figure` / `close_figure`）。**liveplot 對前端無認知**：它只驅動「當前 active backend」，不知道也不偵測誰是 GUI。
   - `backend/__init__.py`：active backend **選擇順序**（與 matplotlib backend 名稱解耦）：
     1. 經 `set_liveplot_backend(backend)`（ContextVar context manager，per-task）註冊者 —— GUI run worker 用它註冊自己的 Qt backend。
     2. `set_default_liveplot_backend(backend)` 設的 process-wide 預設。
     3. 都沒有時，依 matplotlib backend 名稱兜底（`nbagg` → `JupyterBackend`，其餘 → `FallbackBackend`）。
   - 內建 backend（純 matplotlib，**零 gui/Qt 認知**）：`JupyterBackend`（notebook display）、`FallbackBackend`（`plt.subplots` / `draw_idle`）。
   - 對外統一入口（皆 dispatch 到 `active_backend()`）：`make_plot_frame` / `instant_plot` / `refresh_figure` / `close_figure`。
   - GUI 的 backend（`QtLivePlotBackend`）**住在 `gui/app/main/adapters/`、不在 liveplot**：它靠註冊進來，故合法認識 gui（`plot_host`），依賴方向 gui → liveplot。它的 `make_plot_frame` 走 `plt.subplots`（被 GUI custom mpl backend 攔截、attach 進 `FigureContainer`），與裸 `plt.subplots()` 及 analysis figure 同一條渲染路徑；`refresh` marshalling 到主線程；`instant_plot`/`close` no-op（figure 建圖當下已 attach、生命週期歸 container）。
   - `jupyter` 另保留 module-level `instant_plot` / `grab_frame_with_instant_plot`（notebook 動畫特例直接 import 用）。

3. `segments/` — 與 backend 解耦的純繪圖單元（只吃 `Axes`）
   - `AbsSegment`：`init_ax / update / clear`。
   - `BaseSegmentLivePlot`：共用容器與生命週期邏輯。
     - 管理 `segments`、`fig/axs`、`auto_close`、`disable`、`existed_axes`。
     - `__enter__` 會呼叫每個 segment 的 `init_ax()`。
     - `clear()` 呼叫所有 segment 的 `clear()` 後，立即重新呼叫 `init_ax()`，確保 `update()` 之後仍可再次呼叫（不會因為 artist 為 None 而 raise）。
     - `__exit__` 若 `auto_close=True` 且 plotter 持有 figure，則呼叫 backend 的 `close_figure(fig)`。
     - 使用 `existed_axes` 時 `fig=None`（figure 由外部宿主持有，如 `MultiLivePlot` 的 sub-plotter）。此時 `__enter__` / `clear` **不**自行 `refresh()`（守 `fig is not None`）——刷新由持有 figure 的宿主負責。直接呼叫 `refresh()` 仍會 raise（誤用保護：自有 figure 的 plotter 才該自刷）。
     - `disable=True` 時會 early-return，`clear / refresh / update` 都是 no-op。
   - `Plot1DSegment`：多條 `Line2D`，`set_data` 更新，`relim + autoscale_view`。
   - `Plot2DSegment`：均勻格點 `imshow`，以 `set_extent + set_data` 更新，支援 `flip=True`。
     - 若 `vmin`/`vmax` 任一為 `None`，`update()` 會 `autoscale()`；兩者都提供時則固定用 `set_clim(...)`。
   - `PlotNonUniform2DSegment`：`NonUniformImage` 非均勻格點，clim 邏輯同上。
   - `ScatterSegment`：`PathCollection`，`colors` 可作 colormap 值或直接顏色。

4. `plot1d.py` / `plot2d.py` / `scatter.py` — 具體 plotter 包裝層
   - `LivePlot1D` / `LivePlot2D` / `LivePlot2DwithLine` / `LivePlotScatter` 都繼承 `BaseSegmentLivePlot`。
   - `LivePlot2D` 以 `uniform: bool` 選 `Plot2DSegment` 或 `PlotNonUniform2DSegment`，並用 overload 維持型別提示。

## `LivePlot2DwithLine` 特別行為

- `segments = [[segment2d, segment1d]]`（一行兩列）。
- `line_axis`：切線方向（`0` 沿 x、`1` 沿 y）。
- 每次 `update` 會找 `signals` 中最後一條非全 NaN 線為 current line，再往前回溯 `num_lines` 條一起畫。
- 預設樣式：最後一條高亮（`marker='.'`, `alpha=1.0`, `color='C0'`），其餘紅色半透明；可用 `segment1d_line_kwargs` 覆蓋。

## 對外 API 與匯出

- 根模組 `liveplot/__init__.py` 目前匯出：
  - `backend`
  - `make_plot_frame`
  - `AbsLivePlot`、`DummyPlot`
  - `MultiLivePlot`
  - `LivePlot1D`、`LivePlot2D`、`LivePlot2DwithLine`、`LivePlotScatter`
  - `instant_plot`（backend-dispatch；見上「對外統一入口」）。

## 使用模式

```python
with LivePlot2D("freq", "amp") as lp:
    for i, y in enumerate(ys):
        signals[i] = acquire(y)
        lp.update(xs, ys, signals)  # refresh=True 預設
```

多個 plotter 共用 figure：

```python
fig, axs = make_plot_frame(1, 2)
with MultiLivePlot(fig, {
    "a": LivePlot1D("x", "y", existed_axes=[[axs[0][0]]]),
    "b": LivePlot1D("x", "y", existed_axes=[[axs[0][1]]]),
}) as lp:
    ...
```

## 注意事項

- `Plot1DSegment.update` 對 `signals` 做 `.astype(np.float64)`；複數輸入會丟虛部（依 numpy 轉型規則），呼叫端建議先取 `abs/real/imag`。
- `segments/` 內的 matplotlib artist 欄位在 `init_ax()` 前是 `None`。`update()` 先 fast-fail，再把 artist 存到 local 變數後使用；不要在 None check 後反覆透過 `self.im` 等 optional 屬性呼叫方法。
- LivePlot 類若要保留圖（例如後續 `savefig`），請設定 `auto_close=False`。
- 各個單一 segment 的 LivePlot 包裝層（`LivePlot1D`、`LivePlot2D`、`LivePlotScatter`）刻意保留樣板結構，不做公共基底抽象，理由是 `update()` 簽名各不相同，強行統一會犧牲型別提示。
- `active_backend()` 每次呼叫都重新解析（不快取），以保留執行期切換 backend 的彈性。`MultiLivePlot.refresh()` 透過 `backend.refresh_figure()` 呼叫，與 `BaseSegmentLivePlot` 一致。
- GUI 模式下 LivePlot 嵌進 tab，靠的是 GUI run worker **註冊** `QtLivePlotBackend`（`set_liveplot_backend`），非 liveplot 偵測 routing context。liveplot 對 GUI 零認知。
- 純桌面 `qtagg`（非 GUI、非 notebook）跑 LivePlot **刻意走 `FallbackBackend`**（`fig.show` + `draw_idle` 已足夠），非另設專屬 qt backend；要 qt 特化再 `set_liveplot_backend` 註冊即可。liveplot 本身不再有 `qt` backend（已搬成 GUI 的 `QtLivePlotBackend`）。
- `LivePlot` run 與 pyplot analysis **共用同一條渲染路徑**：兩者都經 `plt.subplots`/`plt.figure` → GUI custom mpl backend（`GuiFigureManager`）→ attach 進 `FigureContainer`。worker 端的 `draw_idle` 由 `GuiFigureCanvas` 覆寫 marshalling 到主線程，故不會遞迴建圖。
- Qt bridge 的初始化 thread 很重要；若 bridge 首次在 worker thread 建立，Qt canvas 可能被當成獨立視窗或 attach 失敗，因此 `FigureContainer` 需要在 GUI thread 提前確立 host bridge。

---

## 更新紀錄

| 日期 | Codebase commit | 說明 |
|------|-----------------|------|
| （未知） | — | 初始建立，尚未追蹤更新歷程；下次修改時請補上對應 commit |
| 2026-04-26 | `cd0bc869` | 初次建立更新紀錄（本次全面審閱，內容與 codebase 相符） |
| 2026-04-27 | `5e09cf1c` | 修正 Markdown 結構：合併重複的「更新紀錄」區塊。 |
| 2026-05-19 | `77c6aa2a` | 修正三個 bug（Plot2DSegment autoscale、clear 後狀態損壞、MultiLivePlot 繞過 backend）；補充設計決策說明。 |
| 2026-05-23 | `b81a7a89` | Qt helper 改為委派 `gui.plot_host`；backend 選擇新增 active `FigureContainer` 判斷，讓 GUI 可在 `agg` 下嵌入 LivePlot。 |
| 2026-05-23 | `0843e5bc` | GUI app 引入 custom pyplot backend，但 liveplot 仍保留 helper 路徑；兩者共享 `plot_host` 作為 GUI 宿主邊界。 |
| 2026-05-23 | `4aa97d7a` | plotting routing 改為 task-local `ContextVar[current_container]`；`qt` helper 改依賴 `plot_routing + plot_host`，並強化 GUI thread bridge 邊界。 |
| 2026-06-01 | `69c18922` | `instant_plot` 提升為 backend-dispatch（jupyter display / qt no-op / fallback show），給自建 gridspec figure 用；`BaseSegmentLivePlot.__enter__`/`clear` 對 external-hosted（`fig is None`）plotter 不再 self-refresh（host 負責），修 GUI 下「no own figure」raise。 |
| 2026-06-03 | `c76e5fef` | liveplot 去 gui 認知化：`LivePlotBackend` ABC（`backend/base.py`）+ `set_liveplot_backend`/`set_default_liveplot_backend` 註冊（ContextVar），active backend 選擇改「註冊優先 → 名稱兜底」（不再看 routing container）；jupyter/fallback 改 class、純 matplotlib；Qt backend 搬到 `gui/adapters/qt_liveplot_backend.py`（gui 註冊、方向 gui→liveplot）；統一渲染路徑（run liveplot/裸 plt/analysis 全走 plt→mpl_backend→attach，刪 `create_figure_in_current_container`，跨線程由 `GuiFigureCanvas.draw_idle` 覆寫吸收）。**import liveplot 零 gui 牽連。** |
