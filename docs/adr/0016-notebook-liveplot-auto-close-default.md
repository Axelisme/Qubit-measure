# 0016 — notebook liveplot 用 auto_close=True 預設（不 hack ipympl 避免二次渲染）

**狀態：** accepted（2026-06-03，探索後維持現狀）。
**關聯：** 承 Phase 132 liveplot 去 gui 認知化（[[project_gui_liveplot_decouple_design]]）；起因於用戶在 notebook 跑 LivePlot 時，用 `auto_close=False` 看到「結束後變成兩張圖」，追問能否「圖不關 + 不二次渲染」。

## 背景

liveplot 在 notebook（`%matplotlib widget` / ipympl）的即時繪圖需求：圖要在資料擷取**開始時就顯示**、邊跑邊刷新，而非全部跑完才出現。`instant_plot` 主動 `display(fig.canvas)` 達成這點。

ipympl 在 `%matplotlib widget` 下的行為（實測 ipympl 0.10.0）：IPython 在 **cell 執行結束**時，對命名空間裡**帶 `_ipython_display_` 方法且 figure 仍存活**的 canvas 自動再 display 一次（cell-end auto-show）。於是：

- `instant_plot` 顯示第 1 張（立即，邊跑邊看 ✓）。
- cell 結束、figure 還活著 → ipympl auto-show 第 2 張（多餘）。

`LivePlot` 預設 `auto_close=True`：with-block `__exit__` 時 `plt.close(fig)` 銷毀 canvas → cell 結束無 auto-show → **只 1 張**。`auto_close=False`（GUI 場景用，figure 歸 container 管要留著）在 notebook 下會觸發第 2 張。

## 決策

**notebook 場景維持預設 `auto_close=True`，不引入 hack 去支援「notebook 下 auto_close=False 也不二次渲染」。**

「圖留著不關」是 **GUI 場景**的需求（figure 嵌在 tab container、跑完要留給用戶看，由 `clear_dynamic_canvases` 在下次 run 整批清）。**notebook 不需要這個**：notebook 的圖本就是 cell 輸出，`auto_close=True` 下 `instant_plot` 已讓它即時顯示且邊跑邊刷新，結束 close 只是移除「ipympl 重複 auto-show」的觸發，不影響已輸出的圖。

## 探索過的方案（皆實測，nbclient 無頭執行數 display_data 個數）

目標：figure 不關 + 單張圖 + 後續 update 生效 + 不誤傷顯式 `display`。

| 方案 | figure 存活 | 單張圖 | 不誤傷顯式 display | 結論 |
| --- | --- | --- | --- | --- |
| `auto_close=True`（現狀） | ✗（關閉） | ✓ | — | 可用，但圖不留 |
| B：`Gcf.destroy(manager.num)` | ✗（`fignum_exists`=False） | ✓ | ✓ | figure 半死、脫離 pyplot 管理，危險，否決 |
| D：永久把 `canvas._ipython_display_` 設 no-op | ✓ | ✓ | ✗（連顯式 `display(canvas)` 一起擋） | 有副作用 |
| E：一次性 no-op（顯示後自恢復） | ✓ | ✓ | ✓ | 可行但恢復邏輯繞 |
| **G：`canvas._zcu_shown` 標記 + 包裝 `_ipython_display_` guard**（帶標記時吃掉這次 auto-show 並清標記，之後恢復原行為） | ✓ | ✓ | ✓ | **技術上可行且最清晰，端到端驗證通過** |

G 端到端：真實 `LivePlot1D(auto_close=False)` + G hack → 1 張圖、無 error、figure 存活、update 生效。

## 為何否決可行的 G

G/E 都靠**覆寫 ipympl 的私有協議方法 `_ipython_display_`**。對比 Phase 132 接受的 `GuiFigureCanvas.draw_idle` 覆寫——後者覆寫的是 matplotlib **公開、預期被 backend 定制**的 canvas 方法；而 `_ipython_display_` 是 ipympl 的**私有、版本相關**實作細節（未來 ipympl 版本可能改名/改語意，hack 要跟進）。

為一個「notebook 場景非必要」的需求（圖留著不關）引入一個押在第三方私有協議上的 hack，違反最小驚訝與責任明確。**沒有乾淨的實作方式 → 不引入。**

## 後果

- notebook 用 LivePlot：用預設 `auto_close=True`，`instant_plot` 立即顯示 + 邊跑邊刷新，結束自動 close，單張圖。
- `auto_close=False` 在 notebook 下會雙顯示——這是**已知且接受**的（該參數本是 GUI 場景用）。
- 若未來 ipympl 改掉 cell-end auto-show 行為、或此需求變強，重啟時 G 方案的實測結論在此可直接取用。
