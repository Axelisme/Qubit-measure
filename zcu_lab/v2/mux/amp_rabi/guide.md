# Mux amplitude Rabi core

`MuxAmpRabiExp.run(config, *, context)` 同時掃描選定 qubits 的 drive gain，以 mux 讀出平均 IQ。這是無 GUI 的 core 入口。它不建立 Form、recipe、catalog 或 writeback proposal。

## 輸入與排程

`MuxAmpRabiCfg.modules` 包含共同 reset 與完整 `MuxReadoutCfg`。完整 `tones` 每個元素包含 active、drive 及 sweep，順序就是原始 tone index。只有 active tone 必須提供 drive。Active drives 使用獨立 generator，不能占用 readout generator。

每個 active sweep 的 start/stop 是 normalized gain，必須在 `[-1,1]`。範圍可以不同，點數必須相同。第 p 點同時使用每個 active tone 的第 p 個 gain，不作 Cartesian product。Inactive tone 的 sweep 不限制點數。Core 驗證 numeric cfg，完整 qubit/resonator 綁定由呼叫端負責。

每點 compile 一次，依序執行共同 reset、Join 內的獨立 drive pulses、共同 mux readout。Branches 同時開始，短 pulse 完成後等待長 pulse。Core 不做末端 padding，不改 drive pre/post delay。所有 qubits 共用 reps、rounds、initial_delay 及 relax_delay。Native conversion 留在 program。

## 執行與結果

```python
from zcu_tools.experiment.records import RunRecord
from zcu_lab.v2.mux.amp_rabi import MuxAmpRabiExp

core = MuxAmpRabiExp()
result = core.run(cfg, context=context)
context.cancel_signal.raise_if_error()
source = RunRecord(cfg=cfg, result=result)
outcomes = core.analyze(source, core.Options(), plots=analysis_plots)
```

RunContext 提供相容的 soc/soccfg、caller-owned Plots 與每次新建的 StopSignal。Caller 負責 finish/release 與裝置清理。Run 不保存檔案或上次狀態。

`result.tone_indices` 是 active 原始 indices，形狀 `(K,)`。`coordinates` 是各 drive 自己的量化 normalized gains，`signals` 是 ADC-length-normalized complex IQ。兩者都是 `(K,P)`。`measurement_i` 用該原始 tone 的座標畫 magnitude，不因 label 重複覆蓋圖。

Program 平均 reps/rounds，completed-round hook 保留目前點已完成 rounds 的平均。取消後，未取得點維持 NaN。Caller 先呼叫 `raise_if_error()` 檢查 failure，再用 `cancel_signal.is_set()` 區分取消與完整結果。Acquisition 錯誤不因 partial data 而成為成功。Cfg 與 build 錯誤直接傳出。

## 分析與保存

分析沿用單 qubit undamped Rabi fitter 與 zero-gain phase 政策。Options.skip 先移除前幾個 zip 點，每條曲線再保留有限 IQ，至少需要六點。每個原始 index 回傳 `MuxAmpRabiFit` 或帶原因的 `MuxFitFailure`。Fit 圖命名 `fit_i`。π/π/2 gain 與誤差都是 normalized gain。Fit quality 不改 fitter 或校準政策，數值 fit 成功不代表資料足以校準。

`save_run(source, path, metadata=...)` 保存同次完整 cfg、原始 indices、獨立 gain 座標與 complex IQ。`load_run(path)` 還原 explicit source，不以目前 cfg 補歷史資料。Caller 根據 StopSignal 填寫 complete、partial 或 stopped，不把未取得的 NaN 報為完整成功。
