# `zcu_tools.analysis.fluxdep` 模塊重點文檔

**Last updated:** 2026-06-25（first kernel slice）

本模塊是 Flux-Dependence Analysis 的 notebook-neutral kernel，對應 ADR-0028。它承接 notebook
與 Qt GUI 共用的互動選點、filtering、line selection、one-tone peak detection 規則；adapter
只負責轉譯 UI 事件與渲染容器。

## 範圍

- `processing.py`：頻譜轉實數/正規化、2D peak detection、point downsample、mirror difference。
- `selection.py`：brush selection 的幾何規則，供 grid mask 與 joint point cloud 使用。
- `onetone.py`：one-tone 最大色散頻率、切面平滑、peak detection 與點位輸出。
- `line_picker.py`：半通量/整通量雙線互動狀態機與 mirror-loss 對齊輔助。

## 邊界

第一批 kernel 不包含 database search、search 診斷圖、plotly/matplotlib export 圖、params.json export，
也不負責 GUI/ipywidgets lifecycle。Notebook 舊 module 與 Qt interactive widget 只透過 re-export 或
thin adapter 使用本 kernel。
