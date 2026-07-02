---
status: accepted
---

# ADR-0038 — executor ResultTree + per-node event + measurement bundle contract

關聯 [[0026]]（runner scope 與 stop scope）、[[0027]]（experiment data persistence，明確排除 autofluxdep / overnight workflow collection）。

## 脈絡

`autofluxdep` 與 `overnight` 是 executor-owned workflow：外層先掃 flux 或 iteration，再於每個 outer step 執行多個 measurement leaf。它們和一般 `SignalBuffer` experiment 共用 `Schedule`、`ProgramBuilder` 與 `StopSignal`，但 result ownership 不同：

- 一般 experiment 的 public buffer 是單一 ndarray，`SignalBuffer(on_update=data -> ...)` 每次以完整 array 更新 liveplot。
- executor workflow 的 root result 是 `list[dict[measurement_name, result]]`，每個 leaf 只更新自己的 subtree。

舊 executor buffer 在 root update callback 裡解析 `ScheduleStep.path[1]` 找 measurement name，並在每個 tick 重新 `merge_result_list` 所有 measurement。這讓 plotter 依賴 path layout、讓 flush 語意洩漏到 leaf task，也讓 unrelated measurement update 付出不必要重算成本。

同時，`FluxDepExecutor.run` 與 `OvernightExecutor.run` 各自複製一段 init result、plotter/writer、Schedule scope、measurement init/cleanup、exception/stop、figure close 與 `last_cfg/last_result` lifecycle。task contract 也分裂成 app-local ABC 與 runner-local scaffold，導致 leaf task 共用 API 卻沒有唯一 owner。

## 決策

1. **executor-owned `ResultTree` 實作 `BufferProtocol`**。`ResultTree` 持有 root `list[dict[str, Result]]`，提供 `ResultNode` typed handle、child node、leaf buffer 綁定與 per-measurement subscription。`Schedule` 只看到 `BufferProtocol.data` 與 `trigger_update(step, flush=False)`；一般 `SignalBuffer` 行為不因 ResultTree 改變。

2. **plot update 走 `ResultUpdateEvent`，不解析 path**。`ResultTree` 從 `ScheduleStep.path` 在 buffer 邊界解析 measurement name 與 outer index，向該 measurement node 的 subscribers 發出事件：

   - `measurement_name`
   - `outer_index`
   - `outer_value`
   - `env`
   - `node`
   - `result`
   - `flush`

   Leaf task 的 `update_plotter(...)` 只接 event 與該 measurement 的 stacked result，不再讀 `ctx.path[1]` 或自建 `iteration_index()`。

3. **flush 是 result event 語意**。`ScheduleStep.set_data(..., flush=True)` / `trigger_update(flush=True)` 把 flush 傳給 buffer；`ResultTree` 用 flush bypass subscription throttle。step-level update 只觸發 path 對應的 measurement，root-level `ResultTree.trigger_update(flush=True)` 對所有已訂閱 measurement 廣播。一般 `SignalBuffer` 接受同一 keyword 但維持原本 public `on_update(data)` 形狀與節流行為。leaf task 不再使用 liveplot throttle helper。

4. **`MultiMeasurementExecutor` 擁有 common run lifecycle**。base executor 建立 default outer result、plotter/writer、帶 run env 的 `ResultTree` subscription、`Schedule` scope、measurement init/cleanup、retry、stop handling、writer finish、figure close、`last_cfg` 與 `last_result`。若 caller 直接用 `ResultNode.set(...)` 觸發 subscribed update，`ResultTree` 必須持有 env；否則在寫入前 fast-fail，避免 plotter 才以 `env=None` 爆炸。`FluxDepExecutor` / `OvernightExecutor` 只提供 cfg/env 建立與 outer-loop policy。

5. **runner 擁有唯一 task/bundle contract**。`runner/task.py` 定義 `Acquirer`、`TaskPlotter`、`TaskPersister`、`MeasurementBundle` 與 direct-implementation convenience `MeasurementTask`；`ComposedMeasurementBundle` 可把 acquire / plot / persistence 三段 component 組成 executor leaf。`MultiMeasurementExecutor` 只依賴 `MeasurementBundle`，不依賴 app-local ABC。

6. **flux context 使用 typed tracker**。`FluxDepInfoTracker` 以 `current` / `first` / `last` 三個 `FluxDepInfo` snapshot 取代 dict magic key。必要欄位以 property / `require(name)` fast-fail；`last_or(name, fallback)` 只用於明確 smoothing fallback。未知欄位立即 raise。

7. **executor dependency signatures 明確化**。`FluxDepDeps` / `OvernightDeps` dataclass 不再是 public run input；executor `run` 以 keyword deps 接 `soc`、`soccfg`、`ml`、`predictor` 等穩定依賴，並在 run 內組 typed env。

8. **workflow collection persistence format 不在本 ADR 內統一**。`TaskPersister` 先承接既有 save/load mechanics；`autofluxdep` / `overnight` 的 heterogeneous workflow collection 不納入 [[0027]] grouped Experiment Dataset。若要統一其 on-disk collection format，另立 ADR/task。

## 後果

- executor plotter update 成為 per-measurement event；更新單一 measurement 不重算 unrelated measurement 的 stacked/cache view。
- `Schedule` 的普通 `SignalBuffer` API 維持穩定；ResultTree 只是 executor-owned buffer implementation。
- executor common lifecycle 只住 `MultiMeasurementExecutor`，新增 executor outer-loop 時不再複製 plotter/writer/cleanup/retry 骨架。
- leaf task 可逐步從 direct `MeasurementTask` 遷移為 `ComposedMeasurementBundle`，而 executor 不需要知道 leaf 是 single class 或三段 component。
- typed `FluxDepInfoTracker` 把跨 task dependency 從靜默 `.get(default)` 改成可定位欄位的 fast-fail，降低 flux workflow 內隱相依漂移。

## 拒絕的替代方案

- **保留 root-level callback + path parsing**：短期改動小，但 measurement identity 仍由 `ctx.path` 的 layout 隱式決定，且每次 update 會重新 merge unrelated measurement。
- **讓 `Schedule` 直接認 ResultTree node**：會把 executor workflow concept 塞進一般 experiment runtime。`Schedule` 保持只認 `BufferProtocol`，ResultTree 是 buffer implementation。
- **把 autofluxdep / overnight persistence 併入 [[0027]]**：它們是 heterogeneous workflow collection，不是單一 Experiment Result 的 Dataset Roles；強行套用會混淆 collection identity 與 experiment data identity。
