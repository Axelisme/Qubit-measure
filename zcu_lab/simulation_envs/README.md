# Simulation environments

**Last updated:** 2026-11-02，共用 v2 接線與 tmux／tport

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
