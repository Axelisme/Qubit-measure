**讀者：** lab-author、agent。用 startup 腳本建立或重開六顆 fluxonium 的 entry 與 initial point。

# Six-fluxonium entry kickstart

此腳本只建立或重開 `six-fluxonium/initial`，不連線 SoC、不註冊 device，也不載入模擬環境。

## 執行

在 repo root 執行：

```bash
uv run --directory . --no-sync -- python -m apps.measure \
  --script zcu_lab/kickstart/six_fluxonium/startup.py
```

MCP 啟動新 GUI 時也可指定腳本：

```python
connect(launch="new", script="/absolute/path/to/zcu_lab/kickstart/six_fluxonium/startup.py")
```

首次執行會從相鄰的 `entry-template.yaml` 建立 Q1–Q6、R1–R6、六條局部線 `Q1_bias`–`Q6_bias`、global coil `coil_bias` 與 `MG123456` 的 setup 參數。
腳本接著建立並選取 `initial` point，再保存 aliases：
`mux=MG123456`、`qubit=Q1`、`resonator=$qubit.resonator`。
七條 DC 線都是 `device/current_source` 元件，不會因此連線儀器。

Entry 已存在時，腳本只重開並選取 `initial`，保留元件與 aliases 的修改。
缺少該 point 或 entry 不完整時會報錯，已完成的操作不會自動回滾。
腳本以 GUI 的 project root 存檔，不以 shell cwd 決定資料位置。
需要隔離資料時，啟動指令可加 `--project-root /absolute/path/to/existing-directory`。

## 接線

| 元件／用途 | Generator | ADC／來源 |
| --- | --- | --- |
| Q1–Q6 drive | ch 1–6，依 Q1–Q6 順序 | 無 |
| R1–R6 pulse readout | 共用 ch 13 | 共用 ro_ch 6 |
| MG123456 mux readout | gen_ch 0 | ro_ch 0–5，依 R1–R6 順序 |
| Q1–Q6 局部 DC flux | 無 | 各自的 `Q1_bias`–`Q6_bias` 線 |
| MG123456 global coil | 無 | `coil_bias`，作用於六顆 qubit |

每顆 Qi 引用同序號 Ri，`flux_source` 指向自己的 `Qi_bias`。
Mux group 的 `flux_source` 指向 `coil_bias`。
局部線用來調整對應 qubit 的工作點，global coil 用來調整六顆 qubit 的工作點。
Scalar flux 表單在 Open／Reset 從 `$qubit.flux_source` 取得局部線名稱，不需要 flux 裝置 alias，也不指向 global coil。
`qubit` alias 只選 scalar 表單的目標，不改變 mux group 的六顆順序。
Mux 表單從 group wiring 取得 generator 與各 tone ADC，不從 resonator 的 pulse wiring 取得。

## 待量測的值

接線與示意值沿用 [six-fluxonium 模擬環境](../../simulation_envs/six_fluxonium/README.md) 的配置，但本範本不含模擬器或校準值。
Q1–Q6 的 EJ／EC／EL 使用 GHz，R1–R6 名目頻率為 7000、7100、7200、7300、7400、7500 MHz。
這些數值都是示意，量測後填入實測值。
Group 的 `mixer_freq: 7250` MHz 是示意 LO 設定，使用前須確認硬體 LO 設定並更新。
六顆 qubit 的 `freq` 與 resonator、mux group 的 `time_of_fly` 全部保留 null。
Component 範本使用 `schema_version: 2`。每顆 qubit 的 `flux` 初始為 `{}`，不預填校準。
Global coil 來源仍是 group 的 `flux_source: coil_bias`。
校準後，writeback 向該 qubit 的 `flux.<實際被掃device>` 寫入完整的 `half/integer/period`，不替換其他 entries。
例如 `Q1.flux.coil_bias` 使用 coil device 的原生單位，三欄都必填有限數字，period 非零且可帶負號。
同次接受會另 append 一列原始擬合 record，保存 Run 的 devices 與原校準 frame。
舊 entry 須先經[長期升級入口](../../../docs/parameter-upgrades.md)轉換，startup 不自動升級。

範本不含 module library、readout 校準、qubit pulse 校準或已填好的量測序列。
缺 readout slot 的 mux 表單在 Open／Reset 時，從 group 取得 gen_ch、ro_ch 與 mixer_freq。
Timing 不會因此完成校準，pulse length 等必填空白仍須補齊。
`mux/onetone` 的 trigger 與 pulse length 留白，ADC window 會先填 1.0 us。
`mux/t1` 的 trigger 會先填 0.0 us，pulse length 與 ADC window 留白。
這些表單預設值都不是量測得到的校準值，使用前須確認。
已有 readout slot 時，表單沿用該 slot 的 mixer 與 timing 規則。
修改元件或 aliases 後，要重開或 Reset 表單才會更新初值。

建議順序為 lookback → range_find → mux/onetone → twotone → rabi → T1／T2。
量測前先核對接線與儀器設定，再填入每顆 qubit 的有效參數並連線所需硬體。
