# Simulation environments

**Last updated:** 2026-11-12，envelope 刻度與離線 carrier 工作負載

**讀者：** lab-author、agent。查找 GUI／Notebook 模擬環境、參數與 compile profile 的入口。

物理模型環境包括 [FTF](FTF/README.md)、
[single_fluxonium](single_fluxonium/README.md)、
[integer_fluxonium](integer_fluxonium/README.md) 與
[six_fluxonium](six_fluxonium/README.md)。各目錄提供 startup 與模型參數，
執行 startup 才會組裝環境與資源。[white_noise](white_noise/README.md)
另提供沒有物理模型參數的白噪音範例。
三個物理模型的 builder 位於 `zcu_tools.simulators.<model>.environment`，
YAML loader 位於對應的 `config` package。框架不 import 這些模型。

板子 YAML 由 startup 或範例呼叫端指名，再交給 `program.build_soccfg` 建立 compile config。
各環境目前使用的來源如下。

- single_fluxonium、integer_fluxonium、FTF 與 six_fluxonium 的 startup 共讀
  `profiles/qick_216_v2.yaml`，不在各環境目錄複製它。Generator／readout 型別與
  時脈公式依 `bitfiles/qick_216_v2.hwh` 與 QICK 0.2.436 driver 規則填寫。
  Channel 數量、位置與接線依模擬環境需要配置，不複製實機 inventory。
- white_noise 範例保留相鄰的 `white_noise/profile.yaml`，提供兩個一般 generator
  與一個 ADC，不切到 v2。呼叫時另填 `std_adc`、`seed` 與明確接線或 `routes="all"`。

板子設定獨立於物理模型 YAML。修改 profile 後須重建環境。
模型 YAML 必填 seed、thermal occupation、pruning floor 及既有 numerical controls。
Seed 與 occupation 可以明填 `null`，分別表示 startup 選一次與使用 Bose equilibrium。
每個 receiver 都必填 background period 與 pivot，零 depth／slope 仍關閉效果。
模型 YAML 明填接線，channel 必須存在於 caller 的 board profile。
下表的 ro 是 QICK readout channel，不是 RF converter 名稱。

| 環境 | 模型使用的 generator | 模型使用的 readout |
| --- | --- | --- |
| single_fluxonium | gen0 pulse readout、gen1 charge、gen2 fast flux，皆為 v6 | dyn ro0 |
| integer_fluxonium | gen0 pulse readout、gen1 pump、gen2 Stokes，皆為 v6 | dyn ro0 |
| FTF | gen0 pulse readout、gen2／3 charge、gen7 mux | dyn ro0、PFB ro2／3，依 resonator 順序 |
| six_fluxonium | gen0 pulse readout、gen1..6 charge，七個皆為 v6；gen7 mux | dyn ro0、PFB ro2..7，依 Q1..Q6 順序 |

FTF 的 fast-flux 範例仍是模型 YAML 中的註解，不在預設接線中。
Six 的 v6 drive 不要求額外填 `mixer_freq`；既有 mux mixer 設定保持不變。

共用 v2 YAML 的 gen7 與 dyn ro0／1 共用 tProc wave-output port 7，分別以
`tmux_ch` 0／1／2 分派。Gen0..6 各用 wave port 0..6。
Dyn ro0／1 分別使用 ADC "20"／"22"，PFB ro2..9 共用 ADC "21"。
所有 readout 使用 `tport`，ro0..9 分別接 trigger port 10..19、bit 0。
PFB readout 的 `tproc_ctrl` 與 `tmux_ch` 明填 `null`，不接動態 control port。
Ro0..7 的 readback `tproc_ch` 為 0..7，ro8／9 明填 -1 表示沒有 readback。
各 readout 明填 `b_phase: 32`。

PFB 使用 `axis_pfb_readout_v4`，`decimation: 2` 已由使用者確認，對應
hwh 的 `C_ADC_Decimation_Mode21=2`。Builder 依這些設定推導時脈與 PFB channel 間距。
這份 YAML 不代表已完成板上驗證。tProc pins、`rf.clk_groups` 與 mux generator
的部分必填能力欄位仍保留 compile 佔位值，詳見 YAML 註解。

`profiles/ordinary.yaml` 與 `profiles/mux.yaml` 是獨立板子格式的範例。
前者提供兩個一般 generator 與一個 ADC。後者提供兩個 generator 與五個 ADC，
其中 generator 1 使用 mixmux8，ADC 4、1 依序連到同一 PFB 的 output 0、1。
兩份設定均使用 MHz，明填 QICK 0.2.436 metadata 與 tProc revision 27。
更換 QICK 或板子能力時，由環境作者更新對應 YAML，不由 builder 查環境補值。
Tests 在 `tests/program/profiles/` 自備資料，不讀這裡的設定。

呼叫端先讀 YAML，再交給 program 機制。路徑由呼叫端選定：

```python
from pathlib import Path

import yaml
from zcu_tools.program import build_soccfg

profile_path = Path("zcu_lab/simulation_envs/profiles/mux.yaml")
with profile_path.open() as stream:
    soccfg = build_soccfg(yaml.safe_load(stream))
```

這只產生 compile config，不建立 SoC、不提供 IQ，也不操作硬體。
欄位、時脈推導與失敗契約見 [program README](../../lib/zcu_tools/program/README.md#yaml-compile-profiles)。

## 離線 carrier 工作負載

`workloads.carrier_workload()` 用 `twotone/freq` 的真實 `FreqExp.run` 建立
24 點、21 能階的 carrier 掃描。它呼叫公開 `run_stage` 與模擬 backend，
從 private replay 的公開 `read_replay` 取得 request 與 snapshot，再以
`model_facts` 和 `plan_template` 取得 Engine 輸入。Manifest 記
`program_source: experiment_run`，沒有鏡映實驗的 emit 序列。

Builder 用 Git HEAD 的 `single_fluxonium/default.yaml` 與
`profiles/qick_216_v2.yaml`，不讀使用者工作副本。Carrier 中心是模型的 E1-E0。
掃描範圍是中心 ±100 MHz，沿用 5 MHz 起始 linewidth guess 的 20 倍範圍。
Const pulse 長 1 us、gain 0.1，沿用實驗表單未校準的起始值。
Readout 使用 tracked cavity 頻率與 flight time，gain 0.1、pulse/window 各 1 us。
Reps 與 rounds 各 1，drift 與 pacing 關閉，seed 沿用範本。
Manifest 保存全部 cfg、校準來源、blob hashes、compiled carriers 和 domain span 數。
這些輸入不是實機校準。

以下命令只量這個真實工作負載，不額外重跑合成 inventory。
完整量表可能超過三小時，請先估時並安排 exclusive queue。

```bash
repo="$(git rev-parse --show-toplevel)"
results="$repo/.agent_state/carrier-pareto"
mkdir -p "$results"
uv run --directory "$repo" --no-sync -- \
  python tools/run_heavy.py --log "$results/freeze.log" 1800 -- \
  python -m tools.family_pareto \
  --provider zcu_lab.simulation_envs.workloads:carrier_workload \
  --freeze-inputs "$results/inputs.json" > "$results/freeze-receipt.json"
uv run --directory "$repo" --no-sync -- \
  python tools/run_heavy.py --exclusive --log "$results/run.log" 21600 -- \
  python -m tools.family_pareto \
  --provider zcu_lab.simulation_envs.workloads:carrier_workload \
  --manifest "$results/inputs.json" --reference-cache "$results/reference-cache" \
  --output "$results/matrix.json" --summary "$results/summary.md" \
  > "$results/receipt.json"
```

`carrier_workload(evidence_dir=Path(...))` 另保留建構用 replay 與 live PNG，
供離線驗收判讀。沒有指定時，它讀完暫存 replay 後移除暫存目錄。
Caller 使用完回傳的 numeric source 後呼叫 `close()`。
工具只計時 Engine.evolve，不把建構用 acquire 算進 family 耗時。
通用注入、伴隨 family 證法與 cache 參數見
[量表工具](../../tools/README.md#injected-workloads-and-reference-cache)。

## Envelope gain 刻度

Compiled frontend 的 product 模式將 I/Q table samples 除以 65536。
在這些環境的 v6 generator 上，滿刻度 envelope 的峰值是同 gain DDS 輸出的
`32766/65536 = 0.499969482421875`。
Const、flat-top 平坦段與 mux tone 的 compiled gain 不變。
Gauss、cosine、drag、arb 與 flat-top 兩端使用 product 刻度。
完整係數與韌體出處見 [frontend 契約](../../lib/zcu_tools/program/v2/simulation/README.md#output-amplitude-scale)。

模型 YAML 的 charge、flux 與 incident-field 係數仍以 DDS 輸出為基準，不加補償。
各 entry 範本沒有 envelope π gain 或 pulse library，無須把物理係數翻倍。
舊版模擬中用 envelope 校過的 program gain，等效新值約為舊值的
`65536/32766 = 2.0001220777635353` 倍，另受 gain-word 取整影響。
重新檢查 library pulse 的 gain、waveform 與是否超過 gain 上限。
以 const 校過的 gain 不變；single/six guide 的 const 記錄不受此振幅修正影響。

Integer 環境 field guide 的 STIRAP 與 Raman 是舊 envelope 刻度的歷史量測，
不是目前設定的驗證結果。
STIRAP Pump/Stokes gain `0.015/0.05` 的連續刻度等效值約為
`0.030001831166453/0.100006103888177`。
Raman arb gain 0.9 的等效值約為 1.800109869987182，超過 gain 上限 1。
目前 0.1 us 的設定無法在新刻度重現該 Raman 記錄，不能沿用它的轉移率結論。
詳見 [integer 歷史記錄說明](integer_fluxonium/field-guide.md#envelope-amplitude-revision)。

## FTF startup 與 YAML 設定

GUI 使用 [FTF startup 與 entry 流程](FTF/README.md#建立-entry-與啟動)，Notebook 使用
[single_qubit](../../notebook_md/single_qubit.md)，兩者都直接呼叫公開 `ftf(...)`。
先將 [default.yaml](FTF/default.yaml) 複製為自己的 `ftf.yaml`，再明確指定模型 YAML 與共用的 `profiles/qick_216_v2.yaml`。
工作副本、欄位單位與接線對照由 [FTF README](FTF/README.md#工作副本與調參) 及
[field guide](FTF/field-guide.md) 擁有，schema 與診斷見 [FTF config](../../lib/zcu_tools/simulators/ftf/config/README.md)。
修改 startup 或 YAML 後須重建環境，GUI 操作步驟仍以 FTF leaf 為準。

### 關閉補等與 telemetry

Profiling 時可明確覆寫 `time_scale=0`，不改 numeric settings、RNG、density、readout 或 routes。
以下片段從 repo root 讀取已準備的工作副本與共用 v2 板子 YAML：

```python
from pathlib import Path

import yaml
from zcu_tools.program import build_soccfg
from zcu_tools.simulators.ftf.environment import ftf

configuration = Path("zcu_lab/simulation_envs/FTF/ftf.yaml")
board_yaml = configuration.parent.parent / "profiles" / "qick_216_v2.yaml"
soccfg = build_soccfg(yaml.safe_load(board_yaml.read_text(encoding="utf-8")))
env = ftf(configuration, soccfg=soccfg, time_scale=0)
```

使用完畢後呼叫 `env.devices.close_all_devices()`。若要排除 startup preparation 的計時，先呼叫 `env.wait_ready()` 再啟動 timer。
也可在工作副本設定 `numerics.time_scale: 0`。預設 `time_scale=None` 時 builder 使用 YAML 值。

要一併停用 telemetry，先以 `from zcu_tools.simulators.ftf.config import load_ftf` 匯入 loader，
讀取 `settings = load_ftf(configuration)`，再將上面的建立呼叫改成
`ftf(settings.with_telemetry(enabled=False), soccfg=soccfg, time_scale=0)`。
它回傳設定副本，不改原設定、不建立 telemetry 目錄，並關閉 routing timers 與 telemetry I/O。
YAML 的等效設定是 `numerics: {telemetry: {enabled: false}}`。
`settings.with_telemetry(enabled=True, directory=Path(...))` 選另一個目錄，省略 directory 保留目前設定。
Routing 與 density／readout 數值不變。欄位契約見 [FTF config](../../lib/zcu_tools/simulators/ftf/config/README.md)，
輸出、輪替與失敗政策見 [routing telemetry](../../lib/zcu_tools/simulators/ftf/environment/README.md#routing-telemetry)。

### Workers 與 startup preparation

GUI、Notebook 與直接呼叫的 workers／warmup 選擇，以及 ready 後才計時的操作，見
[FTF 演化與重播](FTF/README.md#演化與重播)。GUI startup 不做預編譯，Notebook 明確啟用。
Framework 的 `EnvironmentWarmup` 介面與 coverage 邊界見
[FTF startup preparation](../../lib/zcu_tools/simulators/ftf/environment/README.md#startup-preparation)。
