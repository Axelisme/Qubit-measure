---
date: 2026-11-18
status: active
superseded_by: null
related:
  - ../adr/0069-experiment-core-and-frontends.md
  - ../adr/0078-evolution-module-boundary.md
  - ../adr/draft/simulation-environments.md
  - 2026-10-16-ftf-simulation-speedup.md
  - 2026-11-17-dop853-speed-route-summary.md
basis:
  repo:
    main: 9171e9234645049674d8721b04b555296c01ae00
    integration: ddb9ba02c77e6204042f9375bbfd50105682ff42
    paired_measurement: 544af11bf874cd66625719a5b962feedf093abcb
  external:
    pyopencl: 2026.1.5
    intel_arc_linux_opencl_driver: 26.40.40196
---

# `dop853` 的 OpenCL GPU backend

## 結論

2026-11-18，使用者選「A」，讓 `dop853` 在 Windows 和 Linux 上提供可選的 OpenCL 執行後端。有可用的 FP64 GPU 時，引擎可把合格的獨立 row 成批送到 GPU，每條 timeline 用一個 work-group。沒有 GPU 套件或可用裝置時，預設仍執行原本的 CPU 路徑，結果逐位不變。公開介面與支援原則見 [ADR-0078](../adr/0078-evolution-module-boundary.md#engine-的-gpu-backend)。

在 Intel Arc 的落地驗收中，24 點、d=32 的 `simultaneous_drive` 用 7 workers。CPU 與 `auto` 在同一次獨占量測中交錯跑三組 ABBA。整次 `Engine.evolve` 的 CPU／`auto` wall 中位比為 **5.677×**，IQR 為 5.527–5.751，高於 2× 的驗收門檻。`auto` 把 24 個 `dop853` row 送上 GPU，24 個 `t4` 尾段留在 CPU。三組的完整 density 最大元素差都是 `9.04e-6`。這是兩種 backend 的差，不是對獨立參考解的誤差。精度驗收仍用完整 density 對參考解 ≤ `2e-3` 的預算。

使用者另選擇讓 private replay 照錄製時的執行裝置重算，並讓 FTF GUI 啟動時在背景做 CPU 準備與 GPU 預編譯。現行條件見[模擬環境契約](../adr/draft/simulation-environments.md)與 [ADR-0069](../adr/0069-experiment-core-and-frontends.md#顯式-cfg-與裝置環境)。

## 問題

FTF 的 sub-GHz 雙 tone 成形掃描要做完整驅動積分。`simultaneous_drive` 的單 worker 盤點中，總時間 9.438 s，`dop853` 獨立 row 就佔 9.394 s，約 99.5%。常數雙 tone 的對照案例走 `multitone_rwa`，沒有相同的瓶頸。受測的成形脈衝每 100 ns 約走 825–885 個 accepted steps。DOP853 每次 trial 要算 12 次 RHS，包含 d×d 複數矩陣乘。CPU 已重用工作空間，但沒有分項量出剩餘時間中矩陣乘佔多少。

GUI 在八核測試機上已用 7 個 CPU workers，保留一核給其他工作。9.438 s 是直接呼叫工廠的單 worker 數字，不能用來算 GUI 加速倍率。要決定是否新增 GPU 依賴，須用相同掃描、相同 family、相同 worker 數成對量。第一次編譯另計，不能混入熱跑倍率。

## 考慮過的選項與比較

### 運算 API 與公開介面

OpenCL 可以用同一套 kernel 照顧 Windows 與 Linux。Intel Arc 原型已能執行需要的 FP64 相位運算，[pyopencl 2026.1.5](https://pypi.org/project/pyopencl/2026.1.5/) 也有兩個平台的 wheel。套件不附 GPU driver。CUDA／CuPy 的 CUDA 路線不支援目標 Intel Arc，ROCm 也不能補上這台機器。[JAX](https://docs.jax.dev/en/latest/installation.html) 的 Intel GPU 路線當時仍是 Linux 上的實驗支援。SYCL／dpctl 可以接 Intel 裝置，但調查沒找到符合這兩個平台與 FP64 需求的驗證。Vulkan 能提交計算，卻不能保證裝置支援 FP64，也仍受 driver 的長工作限制。

使用者說「只有 Windows 和 Linux，其餘採用建議」，接受用 OpenCL 並把 GPU 做成可選依賴。介面曾比較全自動、不自動而只接受明示選擇，以及預設自動但允許指定。最後採 `backend="auto"|"cpu"|"opencl"`。安裝 `gpu` extra 即允許 `auto` 使用 GPU；測試、成對量測與逐位比對仍可鎖定 CPU。明示 `opencl` 則要求能用的裝置。這與 [JAX 的平台設定](https://docs.jax.dev/en/latest/config_options.html#jax-platforms)相似：未指定可退回 CPU，指定而無法初始化就報錯。

`auto` 到第一個合格 row 才載入 pyopencl，找 FP64 裝置，編譯小程式，再用固定案例對 CPU 自檢。探測結果按裝置、driver 與 kernel 版本留在磁碟。找不到可用裝置時，`auto` 記下原因並走 CPU。某維度的程式在執行前無法 build 時，只把該維度留在 CPU。GPU 執行中出錯則直接回報，不能拿重算的 CPU 結果掩蓋。完整契約見 [evolution 的 GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)。

### 平行層級與精度

先比較過讓 host 每步派發小矩陣運算，以及讓整個積分留在 GPU。逐步派發得反覆跨過 host／device 邊界。採用的做法仍在每個 work-group 內平行算一步的 RHS，但讓同一組保有一條 timeline 的積分狀態。排程器在一次 acquire 內收集已就緒的 row，按維度與模型分批送出。每次只送尚未完成的 row。原型有長 dispatch 隨機失敗，沒有證據指向單一逾時門檻。程式改用量到的每批時間，把 dispatch 目標設在約 0.1 s，不讀取或要求修改 Linux 或 Windows 的逾時設定。

全 FP32 的 DOP853 在 d=32、5 µs 樣本對參考解差 `1.688e-3`，已接近 `2e-3` 預算。全 FP64 可壓低誤差，但同一個 5 µs 樣本要 13.10 s。每步起點用 FP64 算 interaction-frame phasor，步內其餘運算用 FP32，只花 3.75 s，誤差 `5.201e-5`，與全 FP64 同級。10 µs 樣本也量到同級誤差。在實際掃描原型中，原始 FP32 density 的 trace 和 Hermiticity 仍過不了下一個 row 的交接檢查。交出前先對稱化並歸一，沒有為 GPU 放寬檢查。

首版只用結構判斷合格與否。取樣資料沒有包絡形狀標籤，knot 數和時長才關係到需要多少步與多少次重啟積分。精度矩陣除了 Gaussian，也補了同結構的非 Gaussian 案例，以及 960 knots、1 µs 的角落。該角落 GPU 對 CPU 多出的誤差為 `1.9e-7`，低於預算一半。960 knots 與 10 µs 同時取上限的組合沒有實測。

### CPU 與 GPU 的分工

同一批量測估得 CPU 約 8.2 row/s，GPU 約 52 row/s。若把同種 `dop853` row 分給 CPU 和 GPU，並假設兩邊同時收工，24 點的 GPU 批次可從約 0.46 s 降到 0.40 s。少約 13% 是這次吞吐估出的上限，不是量到的加速。代價是要準確估計兩邊的工作量，還可能縮小 GPU 批次，因此本次沒做。

使用者另問，能否讓 CPU 在 GPU 忙時算**其他**獨立演化。不同 timeline 相互獨立，但同一條 timeline 的下一段要等上一段交出 density。本次排程先收齊就緒的 GPU 工作，再把續段送回 CPU，兩邊輪流。這和刻意把同種 row 分給兩邊是不同的問題，值得在混合工作負載上另量。

### Private replay

GPU 與 CPU 的 density 不逐位相同。版本檢查若不看執行裝置，GPU 錄的一輪可能在純 CPU 機器默默算出不同結果。討論比較了四案：

| 選項 | 結果 |
|---|---|
| A. 記裝置，但用重算機的預設 backend | CPU 錄製搬到有 GPU 的機器可能被誤拒。 |
| B. 只記 pyopencl 版本，文件不保證逐位重現 | 看不出同一版本的套件是否用了不同裝置或 driver。 |
| C. 錄製 replay 時一律用 CPU | 錄製時失去加速，且路徑與一般 acquire 不同。 |
| D. 記實際裝置，照錄製的裝置重算 | CPU 錄製強制 CPU，不受 pyopencl 是否安裝影響。GPU 錄製要求裝置、driver、OpenCL kernel 與 pyopencl 的身分完全相同，不符就報錯。 |

使用者答「D」。即使 GPU 身分相同，新 Engine 若把第一次編譯算進成本，或採不同 worker 數，仍可能選到不同路徑。設計者代裁，待使用者覆核：CPU／GPU 選路只看 acquire 內容、worker 數與裝置身分。GPU 一律用熱 build 的成本估價，replay 記下並重用錄製時的 worker 數。不記每個 row 的選路，也不重建 Engine 的暖機歷史。

### GUI 啟動預編譯

最初代裁選擇在 GUI 只預編譯 GPU，打算加 `warmup="gpu"`。理由是把 ADR-0069 的「GUI 不預編譯」當成使用者對 CPU 準備的限制。查歷史才發現，這句是 agent 在 2026-10-20 改入，沒有使用者裁決。使用者說：「这个 ADR 0069不预编译的裁决，我怎么没印象呢？我好像没有。…我预期 GUI 启动时会预编译啊。」

重新比較兩案。A 用既有的 `warmup=True`，在背景依序做 23 項 CPU 準備與 GPU 預編譯。B 只預編譯 GPU，第一次 acquire 仍付 CPU 準備的代價。使用者答「A」，希望「使用者体感到的执行模拟速度才不会不一致」。這取代只做 GPU 的代裁，不新增 `"gpu"` 值。Notebook 維持 `True`，直接呼叫 `ftf()` 的預設仍是 `False`。開啟預編譯時工廠不等待，acquire 在 admission 後、消耗 sequence 前才等準備完成。

## 被否決的選項與理由

- 延用[先不用 GPU 的決定](2026-10-16-ftf-simulation-speedup.md)，或等下一輪通用 benchmark 才決定。先前輸掉的是 host 反覆派發小矩陣乘。這次整段積分留在裝置，並在目標掃描上量到收益。使用者在確認差異後選擇加入 GPU。
- 新增 `dop853_gpu` family 或通用 device registry。GPU 執行的是同一套 DOP853 方法。另加 family 會讓診斷和白名單多出一個算法名稱，也讓同 family 的 CPU／GPU 配對比較失去入口。依 [ADR-0078 的黑盒邊界](../adr/0078-evolution-module-boundary.md#黑盒邊界是-simulateevolution)，仍由 `Engine` 對外，不新增 `zcu_tools` 內部的依賴邊。
- 用環境變數選 backend，或只提供全自動模式。前者把選擇藏在行程狀態，後者無法鎖定 CPU。單一具名參數讓測試、量測和逐位比對可以重複同一個選擇。
- 用 CPU DP5(4) 取代 GPU 方案。它每次 trial 只算 7 次 RHS，但同容差下誤差較大，須重新校準。較早的樣本不能證明它能取代在目標掃描上量到的 GPU 收益，留作無 GPU 機器的另一條路。
- 從 knot 振幅猜 Gaussian，或增設包絡形狀標籤。前者會誤判，後者會跨模組改資料格式。兩者都沒有比結構條件提供更好的誤差證據。
- GPU 錄製搬到 CPU 後改用容差比對，或為重現選路而記下逐 row 裝置。前者改掉逐位重算的契約。後者要擴充記錄格式與 Engine 介面，只為保留一次性冷編譯的估價。

## 留下的問題

- Windows 和其他 OpenCL driver 還沒跑過。Linux 的 Intel Arc 實測不能代替平台驗收。
- 冷快取時，GUI 啟動後立刻 acquire 會等全部準備。Intel Arc、d=24、7 workers 的非獨占單次量測中，不預編譯的第一筆為 22.3 s，工廠返回即 acquire 為 76.9 s，其中 74.5 s 在等。等 ready 再按則為 3.3 s，該次背景準備花 104.7 s，CPU 94.3 s、GPU 10.4 s。各組只有一筆且機器負載不同，只能說明等待的位置和量級。還要量熱快取的日常啟動，再評估是否讓 acquire 只等它會用到的項目。
- 24 點掃描的 GPU 批次期間曾量到 3,233 worker-ms 的 CPU 閒置。要不要重疊排程，得再量有其他 CPU 工作的掃描，確認有多少工作能在同一時間開始。
- GPU 的混合精度路徑換成 DP5(4) tableau 後，尚未量到可公平比較的耗時與精度。
- `multitone_gate` 的 GPU 建表原型過不了相位驗證。FP32 與混合精度的 quarter defect 分別為 `1.06e-3` 和 `9.37e-5`，都高於 `1.9e-7` 門檻。這不在本次 `dop853` 的放行範圍內。

## 依據

設計比較使用了工作負載盤點、外部套件調查、精度矩陣與多輪 GPU 原型。落地數字來自上述 integration 的審查、獨占 G3 重跑與 GUI 背景預編譯量測。原始工作檔未入庫。現行契約可查 [evolution GPU backend](../../lib/zcu_tools/simulate/evolution/README.md#gpu-backend)、[FTF startup preparation](../../lib/zcu_tools/simulators/ftf/environment/README.md#startup-preparation)和 [FTF private replay](../../lib/zcu_tools/simulators/ftf/environment/README.md#private-replay)。

G3 不計冷編譯。獨立冷 build 測試只量 `pyopencl.Program.build`，不含裝置偵測與自檢。前次 production 的 d=32 在空 simulator cache、停用 driver cache 時為 0.258 s。重跑時，保留 driver cache 為 0.284 s，停用時為 0.360 s。保留的那次沒有確認命中。全冷的研究原型量到約 2.3 s，和 production 不是同一支程式，不能當成同一組冷熱對照。
