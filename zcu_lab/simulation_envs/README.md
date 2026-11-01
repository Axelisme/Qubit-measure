# Simulation environments

**Last updated:** 2026-11-02，共用 v2 profile 與暫定 PFB 限制

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

- single_fluxonium 與 integer_fluxonium 的 startup 共讀唯一一份
  `profiles/qick_216_v2.yaml`，不在各環境目錄複製它。這份板子有 16 個 generator
  與 10 個 ADC，依 `bitfiles/qick_216_v2.hwh` 與 QICK 0.2.436 driver 規則填寫。
- FTF 與 six_fluxonium 暫讀各自相鄰的 `profile.yaml`。第二步須先讓 builder 與模擬器
  支援共用 control port 的 `tmux_ch` 分流、`tport` 與其餘實機拓撲，才切到 v2。
  第二步開工前仍須取得使用者對 PFB decimation 的答案。
- white_noise 範例保留相鄰的 `white_noise/profile.yaml`，提供兩個一般 generator
  與一個 ADC，不切到 v2。呼叫時另填 `std_adc`、`seed` 與明確接線或 `routes="all"`。

板子設定獨立於物理模型 YAML。修改 profile 後須重建環境。
模型 YAML 必填 seed、thermal occupation、pruning floor 及既有 numerical controls。
Seed 與 occupation 可以明填 `null`，分別表示 startup 選一次與使用 Bose equilibrium。
每個 receiver 都必填 background period 與 pivot，零 depth／slope 仍關閉效果。
Six 的接線也由模型 YAML 明填，channel 必須存在於 caller 的 board profile。FTF 使用四個 generator
和三個 ADC，其中 gen1 配 mux ADC1、2。single_fluxonium 和 integer_fluxonium
各使用板上的 gen0..2 與 ADC0。gen0..2 皆為 v6，ADC0 為 dyn readout。
six_fluxonium 用十四個 generator 和七個 ADC，其中 gen0 配 mux ADC0..5。

共用 v2 YAML 以現有 builder schema 表達 compile 能力，不是完整實機拓撲。
它與 hwh 有五項偏離。

1. Dyn readout 的 control port 連號為 16、17，不表達實機與 mixmux8 共用 port 12
   及 `tmux_ch` 0、1、2 的分流。
2. Readout trigger 使用 `dport`，不是實機的 `tport` 10..19、bit 0。
   Dyn readout 使用 dport 0，PFB 暫用 dport 1，避免兩者的 trigger bit 重疊。
3. DAC／ADC 名稱連號，不沿用實機 DAC 20..23、00..13、30..33 與 ADC 20、21、22。
   每個 PFB output 各有一個 ADC 名稱，不表達八個 output 共用實機 ADC 21。
4. PFB readback 的 `tproc_ch` 連號，不表達實機 readout channel 8、9 的 `tproc_ch=-1`。
5. Dyn readout 不含 `b_phase`，無法表達實機的 32-bit phase 欄位。

PFB `decimation` 暫填 hwh 設計值 2，來源為 `C_ADC_Decimation_Mode21=2`。
這個值待使用者確認，未在板上驗證。由它推導的頻率與 channel 間距也只是暫定配置，
不是已確認的實機值。任何驗收都不依賴 PFB 數值。

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
