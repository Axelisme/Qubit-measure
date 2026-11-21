---
date: 2026-11-21
status: active
superseded_by: null
related:
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/0083-program-macros-and-timelines.md
  - ../adr/0066-operation-lifecycle.md
  - 2026-11-21-ftf-stop-checkpoints.md
  - ../../lib/zcu_tools/program/v2/simulation/README.md
  - ../../lib/zcu_tools/program/v2/simulation/soc.py
  - ../../lib/zcu_tools/program/v2/simulation/mixin.py
  - ../../lib/zcu_tools/program/v2/simulation/binding.py
  - ../../lib/zcu_tools/program/v2/simulation/result.py
  - ../../lib/zcu_tools/simulate/adc/README.md
  - ../../lib/zcu_tools/simulate/adc/raw.py
  - ../../lib/zcu_tools/simulate/adc/_path_sampling.py
  - ../../lib/zcu_tools/simulators/ftf/environment/README.md
  - ../../lib/zcu_tools/simulators/README.md
basis:
  - "本 repo main 1bb899d61b38479fac11a92d5cc33d90dfe28088；既有 raw telemetry 基準 f05b15d9fb61f89f3b5c50271a7876d19d8908c7；EarlyStop owner 候選 4e6432abcc2f7883822f50617737880308826418"
  - "QICK 0.2.436，repo 以 pyproject.toml 與 uv.lock 固定版本"
---

# 當輪必需 raw 的取消檢查與發佈

## 結論

當 Program 為目前 integrated round 需要 raw 時，取消查詢只在該次 materialization 呼叫中借給支援 cooperative cancellation 的 source。`program.v2.simulation` 定義 optional `CancellableRawShots`，以 `materialize_cancellable(reps, avg_level, *, cancel=...)` 與原本兩參數 `RawShots.materialize` 區分。ADC 不 import Program，也不保存 query；沒有此能力的 source 繼續用兩參數方法，由 SoC 在同步呼叫前與交付前查詢。

Required raw 在 round 發佈之前完成。SoC 驗證、detach compact result、抽取並驗證 raw、建好 transport buffers，再做最後 Stop query。Binding 以原 snapshot／execute／pacing elapsed window 計算同一筆 physical clock credit；required raw 驗證和 materialization 不進入該時間窗，最後 query 通過才入帳。Raw telemetry 沿用單次計時，包含 acquire 內失敗的 materialize、不含下游 shot processing；成功的 deferred materialization 可按原 acquire identity 發出補充紀錄。SoC 隨後發布 buffers/source 並前移 index。Stop、`Cancelled` 或 raw 驗證／抽樣錯誤在這個界線之前發生時，保留上一個完成輪，不入帳、不前移，也不把該輪交給 Program。Program 使用獨立可寫副本，成功的 raw 不重抽。

QICK `start_readout()` 是 accumulated transport 的明示 raw demand，也走同一 required-raw 路徑；只有成功才允許該輪 `poll_data()`。QICK 簽名、ADC/channel 順序與 poll 格式不改。直接 `run_round()`、一般 statistics-only acquisition 與已完成輪 getter 保持 compact／lazy，不因本決定改為 eager raw。Stop 在發佈後才出現時，不回滾已發布 transport；processor 或 hook 失敗照常傳出，不建立整輪 stage／commit transaction。

這是 designer 依既有授權的代裁，使用者未覆核。此規則不宣稱 5 秒目標已達成。

## 問題

FTF 先回傳 compact integrated statistics 和獨立 raw source。[`SimulationSoc.run_round()`](../../lib/zcu_tools/program/v2/simulation/soc.py) 已驗證 statistics、替換 source／transport buffers 並前移 index；[`SimulationBinding`](../../lib/zcu_tools/program/v2/simulation/binding.py) 也已完成該輪 physical clock credit。Program 的 [`SimulationAcquireMixin`](../../lib/zcu_tools/program/v2/simulation/mixin.py) 隨後才 materialize required raw，而這段工作已離開 backend `Cancelled` catch。只把 `Cancelled` catch 擴大，無法撤回已發布的 source、buffers、index 或 clock credit。

QICK 的直接 `start_readout()`／`poll_data()` 也是可觀察的 transport seam：若 start 先發布，poll 才同步抽 raw，Stop 或 raw error 就會留下未成功交付的輪。兩端的可寫 raw arrays 也不能共用同一份 storage，否則 Program 的 `get_raw()` 寫入會修改 SoC 的 transport 資料，反之亦然。ADC 的 [`ADCRawShots`](../../lib/zcu_tools/simulate/adc/raw.py) 與 [physical shot sampler](../../lib/zcu_tools/simulate/adc/_path_sampling.py) 擁有實際逐 shot 工作，不應反向依賴 Program。

在真實 `chain/amp_rabi` 的第一個 gain（−0.3）單點 caller 上，使用原 form 第一點、`gain.expts=1`、100 reps、1 round、seed 123，以及只把 `low-snr.yaml` 副本的 `projection_weight_floor` 從 `1e-9` 改成 `1e-5` 的離線樣本中，raw materialization 為 5.032161 秒；兩次取消查詢間隔為 5.035913 秒。首個 event 為 4.940485 秒。該次沒有送 Stop。樣本有 1 點、554 outcomes、24 states、1 ADC、16 reads、每窗口 304 samples。Low-snr 與 FTF-demo 的 DC bias 不同；輸入和尺寸不代表 51 點首輪或 floor `1e-9`。

## 考慮過的選項與比較

| 選項 | 做法 | 優點 | 代價與失敗邊界 |
| --- | --- | --- | --- |
| 保持現狀 | raw 仍在 publish 後由 Program materialize。 | 不改 code。 | 必需 raw 期間沒有合作檢查；中斷後 raw 會逸出原 catch，SoC 已發布的 source、buffers、index 和 clock 不代表 raw 成功。 |
| A：optional cancellable raw，加 required-raw 發佈界線（採用） | Program 按當輪 demand 明示 `require_raw`；source 可選擇實作 `CancellableRawShots`。SoC 在 raw 成功、驗證與最後 Stop query 後才入帳並發布。QICK `start_readout()` 同樣明示需要 transport raw。 | 不需要的輪保持 compact；只有具合作取消的 source 內部查詢；舊兩參數 source 繼續工作；已完成 getter 不帶過期 query；clock window、raw telemetry 和 QICK 介面可保留。 | 增加 optional protocol、per-round demand 與 publish handoff；必須複製 writable buffers。source 沒有 optional 能力時，Stop 只能在同步呼叫前後被看見。新的檢查點不保證每段在 5 秒內。 |
| B：所有 raw source 共用新增的 `cancel` keyword | 修改 `materialize(reps, avg_level, *, cancel=None)` 並遷移每個 source／wrapper。 | 少一個 optional protocol。 | White-noise、Six wrapper 與不需合作取消的來源也得改；舊兩參數 source 若未更新會 TypeError。用 TypeError fallback 可能把 source 內真錯當成相容性問題。影響面超過目前需求。 |
| C：只把 raw 放進 catch | 擴大 mixin 的 `Cancelled` catch，raw 中止時回報停止。 | 改動較小。 | SoC 在 raw 前已公開新 source／buffers、前移 index 並記 clock；catch 不能恢復前一輪。若清空又會丟失先前完成輪。 |
| D：把 query 留在 FTF adapter 或 raw source | source 持續保存 Stop token，或為 FTF eager materialize raw。 | FTF 內看起來可直接接上 query。 | Retained source 可能被後續 getter 使用，過期 query 會污染已完成輪；eager raw 會破壞 statistics-only 捷徑。另建即時／延遲 source 也會重複管理 identity 與輸入生命週期。 |
| E：完整 round stage／commit transaction | 所有 transport、clock、Program processor、hook 與 aggregate 都先留在 pending 狀態，全部成功後再 commit。 | 可以試圖把 processor error 也納入 transport rollback。 | 需要新的 stage／commit API 或 callback，更多直接 caller 與 QICK adapter 需改；reducer 可能已有副作用，不能承諾回滾。現有契約只要求中斷或 raw 失敗時不發布，不要求 processor failure 回滾。 |

採用 A 但明確限定為 required raw 的交付邊界，不把已發布 transport 變成等待整個 host processor 成功的交易。當輪 demand 由 Program 提供；直接 QICK `start_readout()` 以累積 transport 語意提供 demand；stats-only `run_round()` 仍是 compact。已完成輪的 deferred getter，包括 round hook 內的 getter，不攜帶該輪取消查詢。

## 被否決的選項與理由

- **保持現狀。** 已量到 required raw 的查詢盲區；中斷不會自動讓先前發布的 transport 或 clock credit 消失。
- **B，所有實作者改用同一個新 keyword。** 目前只有部分 sources 需要在自身逐 shot 工作中合作檢查。擴大簽名會遷移 white noise、Six wrapper 與既有兩參數 source，沒有目前 caller 的理由支持這個範圍。
- **C，只補 catch。** catch 只翻譯例外，不能撤銷已替換的 buffers、已發布 source、index 或 clock credit；catch 若包進 processor 又會把真實 processor error 誤當 Stop。
- **D，把 live query 存進 source／adapter。** Query 的生命週期屬於當次呼叫。完成輪可在稍後獨立 materialize，source 不能保留舊的 Stop token；一律 eager 也會移除 statistics-only 捷徑。
- **E，完整 pending transaction。** 該方案超出 raw 交付問題，須重新定義直接 `run_round()`、getter、clock、processor、hook 和 reducer 的可見提交點；reducer 的副作用未必可逆。
- **只改 backend exception catch 而不改發佈順序。** 這是 C 的子集，仍不保證取消輪不會從 SoC getter、poll 或 clock 中漏出。

## 留下的問題

- 既有單點 run 沒有送 Stop，5.032161 秒 raw 與 5.035913 秒查詢間隔只證明改動前有盲區，不證明新檢查點的間隔或開銷。
- 51 點首輪與 floor `1e-9` 尺寸未覆蓋。單點資料不得外推到一般尺寸或 GUI retained-cache raw。
- 新查詢點是 cooperative checkpoints，不會中斷單一同步工作。若要達到放點目標需要改算法速度、調整 RNG 批次或改啟動等待，需另行裁決；不把 5 秒變成 gate。
- Processor／hook 在 transport 發布後失敗仍讓錯誤原樣傳出，不回滾 transport。若未來有既有 caller 要求該部分也具 transaction 語意，需另行設計與驗證。
- 不操作實機；實機 acquire 與 decimated Stop 只維持既有行為。

## 依據

- [ADR-0078](../adr/0078-evolution-module-boundary.md) 已定下 FTF 使用合作式檢查點、5 秒是正常參數下的放點依據而非 gate，也定下 ADC 生成 raw、Program 擁有 `RawShots` 介面。此 ruling 遵循 raw 與 statistics 的 owner 分離。
- [ADR-0072](../adr/0072-module-boundaries.md) 將 `simulate` 定為封閉子模組。ADC 結構符合 Program protocol，不 import Program；取消 query 與 `Cancelled` 維持在 `simulate.boundary`。
- [ADR-0083](../adr/0083-program-macros-and-timelines.md) 維持 QICK round 入口、`SimulationSoc` per-acquire query 及 raw 介面在 Program。此 ruling 遵循既有入口，不改 QICK signature。
- [ADR-0066](../adr/0066-operation-lifecycle.md) 區分 cancellation 與真實失敗。此 ruling 只讓發佈前的 Stop／raw cancellation 留下上一個完成輪，不把 processor 或 hook failure 改成成功取消，也不擴成全 round rollback。
- [Simulation acquisition README](../../lib/zcu_tools/program/v2/simulation/README.md) 與 [simulator acquire telemetry](../../lib/zcu_tools/simulators/README.md#acquire-telemetry) 保留 `materialize_wall_s` 的單次計時、acquire 內失敗耗時與 deferred getter 的補充紀錄；這個決定不新增或重複計時。
- [FTF Stop：合作式檢查點與取消傳輸](2026-11-21-ftf-stop-checkpoints.md) 記錄使用者 2026-11-21 的原話：「3C，但設計上5秒內」與「我认为能及时中断也是一个挺重要的功能。」這兩句支持合作式 Stop 與放點目標，不代表新方案已達標。
- RawShots 放在 Program 而非 core 的既有使用者確認原話是：「基本認同丙，但我質疑有需要進core嗎？」對將介面放在 program 的方案答「同意」。
- 零輪停止的使用者裁定原話是：「好吧，那就先保留。但这个东西应该属于 early stop mixing 的内容，而不只是模拟器或者是一些其他的模块的内容。其他模块应该只是利用这个机制而已。」本次延續由 `EarlyStopMixin` 統一收尾的 owner，不讓 SoC 重做零輪判定。
- Required-raw protocol、clock 與發佈順序由 designer 依既有授權代裁，**使用者未覆核**。原因是現有程式已能指出單一發佈接縫，角度調查與獨立 challenge 沒有提出需要新增使用者方向的矛盾；若更短查詢間隔必須改算法或啟動行為，仍須另行取捨。
- 離線收據未入庫。單點、輸入副本、原 query gap、Stop 點與輸出比對的必要事實已寫在本文；尺寸限制亦在「留下的問題」中明列。QICK 對應版本固定為 0.2.436。
