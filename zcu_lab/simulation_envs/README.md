# Simulation environments

**Last updated:** 2026-11-01，caller-owned compile profiles

**讀者：** lab-author、agent。查找 GUI／Notebook 模擬環境、參數與 compile profile 的入口。

各環境目錄擁有自己的 startup 與模型參數。現有環境包括
[FTF](FTF/README.md)、[single_fluxonium](single_fluxonium/README.md) 與
[six_fluxonium](six_fluxonium/README.md)。執行 startup 才會組裝環境與資源。
每份 startup 讀取同目錄的 `profile.yaml`，再用 `program.build_soccfg` 建立板子。
板子設定獨立於物理模型 YAML。修改 profile 後須重建環境。FTF 使用四個 generator
和三個 ADC，其中 gen1 配 mux ADC1、2。single_fluxonium 和 integer_fluxonium
各用三個 generator 和一個 ADC。six_fluxonium 用十四個 generator 和七個 ADC，
其中 gen0 配 mux ADC0..5。

`white_noise/profile.yaml` 提供白噪音範例的板子，呼叫時另填 `std_adc`、`seed` 與
明確接線或 `routes="all"`。它沒有物理模型參數。見 [white_noise](white_noise/README.md)。

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
