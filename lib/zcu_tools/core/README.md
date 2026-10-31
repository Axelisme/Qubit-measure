# zcu_tools.core — shared stateless contracts

**Last updated:** 2026-11-01

**讀者：** framework-dev。跨模組共用的無狀態資料與格式契約。

`core` 提供 resources 與 program 共用的無狀態契約。Repo 內只依賴 utils，不持有 GUI、資源 scope 或硬體。

- `script_protocol` 定義 Question／Answer 與 recipe 使用的 OperationStatus。Question 共用 answer model 推導、wire 驗證與 evidence-dependent validation。Recipe 與 workflow 共用 async host，由 host 持有 Future 與控制狀態。Recipe 的 Analyze／Ask 公開 await 回非 Optional，無結果取消由便利層的普通 RecipeCancelled 表達。Workflow 使用的 `Completed(value, facts)`、作者分析用 Failed 與 OperationFacts 由 experiment.operations 擁有。`value` 是回傳值，`facts` 是當次 operation 的事實。`facts.stopped` 只記錄 operation 自身在結束時的 stop bit，不判斷資料完整度。只取消 operation 的子孫不會把它標成 stopped。Core 例外原樣傳出。
- `complex_codec` 擁有唯一 `{"__complex__": [re, im]}` scalar 格式，供 GUI 與 resources 共用。Schema 決定有限值限制，expression 決定數值語意。Resources view 將 scalar tag 讀為 native complex，手動 scalar 寫入編回 tag。Snapshot、document_value、document_fields 與 ParameterContext 等 non-view 路徑保留 raw tag。
- `element.ElementIdentity` 是 captured actual、estimate 與 writeback candidate 共用的原始 list index／同次 name。它只作報告標識，不解析 alias、participants 或寫入目的地。
- `numeric_scalar.normalize_numpy_scalar` 將單一 numpy floating／integer scalar 投影為 Python float／int，供 AnalysisDocument 與 AcceptedWrite 共用。它不遞迴；finite 與 strict JSON 驗證留在各邊界。
- `simulation` defines immutable numeric acquire/round requests, snapshots, pulse events, readout windows and joint ADC signals. `ChargeDriveRequirement(name, gen_ch)` carries caller-supplied charge-line requirements; environments own physical validation. Templates store per-point event/readout arrays without expanding repetitions; `materialize_point` expands one diagnostic timeline. Integrated round statistics contain only counts, conditional moments and totals. They contain no callback or raw generator. Environments separately supply program-owned RawShots; raw and direct totals have independent realizations with the same expectation. `SamplingIdentity` owns the common local PCG64 seed/stream format. Times are integer ticks in the request's TimeBase, frequencies MHz, phase degrees and gain normalized. It has no driver, Module or GUI dependency. Ownership and shapes are documented in [program simulation](../program/v2/simulation/README.md).
- `simulation.TimeBase` owns the rational common clock, exact fixture grid checks and scalar/int64-array conversion to correctly rounded ns. Program supplies clock frequencies without passing a driver. Requests and the compiled frontend now use integer ticks. TimeBase also converts aggregate integer ticks to seconds for pacing without applying the per-request ns bound. Physical consumers combine integer endpoints and sample grids before calling TimeBase.ns. Its optional integer quantum_ns applies exact ties-even support rounding. Private replay v2 stores the rational clock and tick arrays under ADR-0084.
- `waveform_format` 擁有 NPZ samples、opaque JSON recipe、codec 與單一格式錯誤。它不讀寫檔案，也不解讀公式。
- Resources 負責 formula 語意、資產命名及檔案生命週期。Program 與 sim 播放 path 指定的 samples。
- 配置模型由 `cfg_model` 擁有，格式版本由頂層 `format_version.py` 擁有。
