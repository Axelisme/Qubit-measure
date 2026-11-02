# v2 experiment authoring

**讀者：** lab-author、framework-dev、agent。撰寫實驗 core，查閱 records、資料形狀、執行與保存慣例。

**Last updated:** 2026-11-02，查表掃描與非均勻 delay 用法

使用者入口見 [tutorial](../tutorial/README.md)。

## 目錄

- [實驗序列與 recipe 撰寫規則](#讓使用者只看一個檔案就能理解實驗)
- [寫法慣例](#寫法慣例)
- [使用者實驗的測試](#使用者實驗的測試)
- [實驗家族 records](#實驗家族-records)
- [目錄佈局](#目錄佈局)
- [實驗特有的分析與保存](#實驗特有的分析與保存)
- [典型 Exp.run 範本](#典型-exprun-範本以-onetonefreqcorepy-為例)
- [Timeline program 與 Schedule 取數](#timeline-program-與-schedule-取數)
- [v2 cfg 組合慣例](#v2-cfg-組合慣例)
- [Signal/Raw 處理慣例](#signalraw-處理慣例)
- [v2 工具與實驗輔助](#v2-工具與實驗輔助)
- [Lookback 與 Tracker](#lookback--tracker)
- [Twotone RB 測量策略](#twotone-rb-測量策略)
- [外部中斷支援](#外部中斷支援cancel_flag)
- [新 experiment 檢查清單](#寫新-experiment-時的檢查清單)

`zcu_lab/v2/` 擁有使用 program/v2 的具體實驗。每個實驗的 `core.py` 保存 cfg、Result、run 與分析政策。共同實驗介面、Result 保存映射與 cfg 組裝見[框架 README](../../lib/zcu_tools/experiment/README.md)。本頁保留實驗家族、具體 workflow 與撰寫慣例。

GUI adapter 的 module writeback 直接提出持久 ModuleSpec，由共用 Form 編輯，再交 Library owner 保存。Core 仍接收執行用 typed cfg，不依賴 GUI 表單或 draft。作者入口見 [measure authoring](measure-authoring.md)，跨宿主契約見 ADR-0065、ADR-0008。

一般實驗以 [runtime](../../lib/zcu_tools/experiment/v2/runtime/README.md) 的 `SignalBuffer` 與 `Schedule` 編排 host loop，將 Timeline program 交給 `run_acquire*` 取數。Runtime 的 buffer、stop、retry 和 lifecycle 機制見其文件。

## 讓使用者只看一個檔案就能理解實驗

以下四條規則適用於 `zcu_lab` 的實驗與 recipes。序列與流程由各實驗自己寫出，
不要為了減少相似程式碼，把使用者要讀的步驟藏到共用 helper。

### 1. 在 core 直接寫出 pulse 序列與分析政策

core.py 的 Experiment 擁有 run 與 analyze 的計算；gui.py 只接線、解析來源、決定 writeback 與 operator guide。
擬合、逐 tone 迴圈、失敗隔離與分析結果型別都歸 core。GUI 為 writeback 做的有限值檢查與摘要整理不算分析計算。
mux/onetone 的逐 tone 數值計算、outcomes 與單 tone 失敗隔離由 core 擁有。
gui.py 保留來源與 route prior 解析，以及 writeback。

`core.py` 的 `run()` 直接列出 pulse、delay、fork／join 與 readout 的先後關係。
共用 helper 只做驗證、資料整形、平均等機械工作，不用 mode 或旗標切換不同實驗。
兩個實驗可以共用平均工具，但各自在 `run()` 寫出自己的序列。

例如 [mux Amp Rabi](mux/amp_rabi/core.py) 每個掃描點先 reset，
所有參與 drive 同時起跑，最後共同 readout。以下是序列片段，省略準備與取資料：

```python
builder = TimelineBuilder(
    context.soccfg, reps=config.reps, relax_delay=config.relax_delay,
)
builder.emit(Reset("reset", reset))
timelines = [f"drive_{index}" for index in scan.indices]
builder.fork("default", into=timelines)
for index, pulse, timeline in zip(scan.indices, pulses, timelines, strict=True):
    builder.emit(Pulse(f"drive_{index}", pulse), timeline=timeline)
builder.join(*timelines, into="default")
builder.emit(MuxReadout("readout", readout))
```

短 branch 等最長 branch 結束才 readout，不把短 pulse 的末端對齊到最長 pulse。
Ramsey 與 Echo 也各自列出 pulse 與 delay，不能交給 `acquire(mode="ramsey")` 選序列。

同一條規則也適用 `analyze()`。Echo 在自己的 core 明示跳過第 1 點，Ramsey 在自己的
core 依資料選固定 0°／180° 相位。共用 fitting helper 只接收明確的 skip、phase 等
擬合參數，不用實驗名稱或 mode 選擇不同實驗的政策。phase 只傳數值或代表自由擬合的
`None`，不傳選政策的 callback。各實驗獨立完成分析與畫圖，只共用數值分析工具。
例如 mux GE 對每個 tone 呼叫 `singleshot_ge_analysis`，自己用 matplotlib 畫 `fit_i`，
不組假 RunRecord 呼叫單 qubit GE 的 analyze。

Workflow 的壽命模型比較留在 workflow，mux T1／T2echo core 只算本次選定的模型。
Mux onetone core 以 `Options.edelay_seeds` 接收每 tone 的種子。Adapter 決定手動值、
capture prior 與全域搜尋的優先序，並記錄來源與決定 writeback，不把 route prior 政策塞進 core。

### 2. 在 recipe 函式本體寫出操作流程

Recipe 本體依序開 tab、選 module、逐欄 set、依序套 overrides、check_ready、Run、Analyze，再詢問 writeback。
來源選擇與掃描端點直接留在主體，helper 只共用數字呈現或逐 Q 分析 updates 等機械對應。

例如 [mux Amp Rabi recipe](../recipes/mux_amp_rabi.py) 應直接讀得到
`session.open_tab(...)`、`tab.select(...)`、`tab.set(...)`、
`await tab.run()`、`await tab.analyze("primary", result=T)` 與
`await tab.ask_writeback()`。接受答案後才 `tab.accept_writeback(decision.items)`。
Analyze 的例外直接交給作者，包含 fitting 例外；作者決定是否捕捉，未捕捉就使 recipe failed。每次 Analyze 後先檢查 `analysis.result`，因為子層分析取消且無結果時仍可返回 result=None、status="failed"。這個取消分支回報 `analysis.reason`，不 Ask、不接受 writeback、不跑相依 Post，不用來處理 fitting 例外。

Run 留在 try 外，成功返回後的流程以既有 finally 同步 `tab.save()`。根取消使下一個 await raise 普通 ScriptCancelled，進行中的 Run 先 drain 並保留 facts。該 Run await 被根取消時不進後續 Save finalizer，partial 留在 tab 交 agent 決定。作者不新增取消分支或 Save finalizer。
Inputs 繼承 RecipeInputs，框架只作 pydantic／MCP JSON schema 驗證，不渲染 recipe inputs Form。
可省略欄位用 KEEP，null 原樣交 owner。Read 與寫入都用 dotted string。
明填 scalar 前直接 command to_inline，保留 waveform family。已 inline 時不改值。
完整宣告、Analyze 例外與取消契約見 [recipe 教學](../tutorial/04-recipe.md)。

### 3. Docstring 說明量測、序列與結果

`core.py` 的 module 與公開函式 docstring 只寫序列、輸入、結果與失敗條件。
開頭先說量什麼、序列怎麼做、回傳結果代表什麼。操作教學、使用範例與環境指引放該實驗的 `guide.md`。
單位、資料形狀及必要的失敗條件仍要交代。Lifecycle、rollback 與 caller 責任
只在必要時用一句帶過，詳細契約連到 framework 文件，不在每個實驗重貼。

例如 [mux Ramsey](mux/t2ramsey/core.py) 的 `run()` 可用這樣的量測說明開頭：

```python
"""Measure Ramsey fringes for the selected qubits with shared mux readout.

Apply simultaneous x90 pulses, wait the requested delay, then apply the
final x90 pulses and read out together. Return requested delays in us
and mean complex IQ per participating tone; the fringe decay measures T2.
"""
```

這裡的 IQ 是量測結果，T2 由後續分析取得，不把擬合值說成 `run()` 的回傳值。

### 4. 用註解與空行分開階段

在準備、序列、取資料與平均等階段之間留空行。註解說明這段做什麼、為什麼，
不逐行重述賦值或函式名稱。程式註解用英文。

例如 [mux Amp Rabi](mux/amp_rabi/core.py) 的階段可以標成：

```python
# 1. Prepare selected tones without changing their original readout indices.
...

# 2. Start drives together; read out after the longest branch finishes.
...

# 3. Collect mean IQ and restore tone-major result order.
...
```

這是分段示意，不是可執行的範本。Recipe 同樣按設定、量測、分析與 writeback 問答分段。

## 寫法慣例

實驗使用以下存檔軸、分析選項、來源選擇與繪圖寫法。
序列、獨立分析與 docstring 的規則及短例見[前節](#讓使用者只看一個檔案就能理解實驗)。

### Cfg 名稱、順序與取數欄位

掃描欄位用被掃量的單數名，例如 cfg 的 `delay`、`gain`、`freq`。
`declare_sweep`／`add_sweep` 名稱跟 cfg 欄位相同，Result 用 `delays`、`gains`、`freqs`。
新實驗的軸 label 沿用同一個字，例如 `Delay`。既有存檔軸的 label／unit 不改，避免舊檔無法載入。
Module 名稱等於提供設定的 cfg 欄位名，例如 `Pulse("pi_pulse", cfg.pi_pulse)`。
欄位名寫角色，例如 `stark_tone`、`probe`，只有角色對稱時才用 `gain1`／`gain2`。
既有持久 cfg 欄位改名時，隨遷移更新 `cfg_schema_version`。

Cfg 宣告順序就是 Form 顯示順序，依序放播放順序的 modules、外到內的掃描量、取樣方式、取數欄位。
Form 預設不寫 `field_order`。特殊需求才覆寫順序，並在原地用一行註解說明原因，
例如讓 mux 的 participants 出現在 tones 前。

取樣方式用 `sampling: Literal["linear", "decay"]`，掃描欄位只接受 `SweepCfg`。
Linear 使用線性格點，decay 把同一時間窗的點集中在衰減段，不另接受 list 模式。
各實驗保留原本的取樣預設；既有 `uniform` 欄位待遷移時改名。

取數欄位在每個 cfg 最後完整寫出，不用額外基底或型別別名藏住欄位。
平均類建議提供 `reps`、`rounds`、`relax_delay`，有 SNR 定義才加 `earlystop_snr`，
需要才加 `initial_delay`。這是風格建議，不因缺少標準組的某一項阻擋 review。
只有超長時間掃描才加 `acquire_retry`。目前開放 `twotone/fluxdep` 與
`singleshot/t1/t1_with_tone_sweep`，其他實驗不加。Workflow 的 retry 從自己的 settings 傳入取數 helper，不放進 core cfg。

Readout 選實驗接受的最窄型別。只接受 pulse 的寫 `PulseReadout.Cfg`，只接受 mux 的寫
`MuxReadout.Cfg`，確實支援兩者才寫 `Readout.Cfg`。其他 module 同樣用自己的 `.Cfg`。
以下是新實驗的 cfg 宣告片段，省略 run 與 Result：

```python
from typing import Annotated, Literal

from zcu_tools.cfg_model import ConfigBase, FieldMeta
from zcu_tools.program.v2 import Pulse, PulseReadout, Reset, SweepCfg


class DelayCfg(ConfigBase):
    reset: Reset.Cfg | None = None
    pi_pulse: Pulse.Cfg
    readout: PulseReadout.Cfg

    delay: Annotated[SweepCfg, FieldMeta(label="Delay", unit="us")]
    sampling: Literal["linear", "decay"] = "linear"

    reps: int = 1000
    rounds: int = 10
    relax_delay: Annotated[float, FieldMeta(unit="us")] = 1.0
```

Module 建構時可用 keyword 覆寫數值或 sweep，不修改呼叫端 cfg。
`None` 表示沿用 cfg。以下例子在 builder 上建立頻率 sweep，再交給 Pulse：

```python
from pathlib import Path

import yaml
from zcu_tools.program import build_soccfg
from zcu_tools.program.v2 import SweepCfg, TimelineBuilder
from zcu_tools.program.v2.modules import ConstWaveformCfg, Pulse

pulse_cfg = Pulse.Cfg(
    ch=0, nqz=1, freq=100.0, gain=0.2,
    waveform=ConstWaveformCfg(length=0.1),
)
profile_path = Path("zcu_lab/simulation_envs/profiles/ordinary.yaml")
soccfg = build_soccfg(yaml.safe_load(profile_path.read_text()))
builder = TimelineBuilder(soccfg)
freq = builder.add_sweep("freq", SweepCfg(start=100.0, stop=102.0, step=1.0, expts=3))
builder.emit(Pulse("probe", pulse_cfg, freq=freq))
program = builder.build()
```

Pulse 支援 `freq`、`gain`、`phase`、`length`。PulseReadout 支援 `freq`、`gain`、`length`、`ro_length`。
Reset 各變體的參數見 [module 建構參數表](../../lib/zcu_tools/program/v2/modules/README.md#constructor-overrides)。
不支援 length sweep 的 waveform 仍會報錯，MuxReadout 不接受覆寫。
`ro_length` 不支援 `TableParam`，建構 `PulseReadout` 時 raise NotImplementedError，不會進入 build。
線性 ADC window sweep 保留 QICK 的既有編譯限制。
既有 `set_param` 仍可用，但它會修改呼叫它的 cfg，應只用在自己持有的 cfg 副本上。

### 存檔軸用關鍵字，固定座標不塞進 Result

`Axis`／`Z` 在欄位名之後用關鍵字寫 label、unit、scale 與 dtype。
Scale 表示 `disk = memory * scale`，axes 仍是 inner-first。
固定類別軸用 `values`，不要求 Result 有對應欄位：

```python
import numpy as np

from zcu_tools.experiment import AxesSpec


frequency_axis = AxesSpec.Axis("freqs", label="Frequency", unit="Hz", scale=1e6)
class_axis = AxesSpec.Axis(
    values=(0, 1, 2), label="Population", unit="None", dtype=np.int64,
)
signal = AxesSpec.Z("signals", label="Signal", unit="a.u.")
```

`field_name` 與 `values` 必須恰好給一個，否則建構時 raise ValueError。
Z 的每個維度仍須匹配軸長，固定軸也計入。固定值不會在 load 時寫進 Result。
`GroupedAxesSpec` 的 `VariableAxisSpec` 全部使用關鍵字。
`VariableZSpec` 可用位置參數指定 field_name，label、unit 與其餘選項使用關鍵字。
`VariableAxisSpec.generated_arange(label=..., unit=...)` 也使用關鍵字。

### 分析選項自行驗證，fit 結果用屬性讀取

分析選項以 pydantic dataclass 的欄位限制驗證，不在 gui.py 重做同一檢查。
例如負的 skip 在建構時就拒絕，GUI 顯示表單錯誤，remote 回 `INVALID_PARAMS`：

```python
from typing import Annotated

from pydantic import Field
from pydantic.dataclasses import dataclass

from zcu_tools.cfg_model import FieldMeta


@dataclass(frozen=True)
class AnalyzeOptions:
    skip: Annotated[int, FieldMeta(label="Skip leading points"), Field(ge=0)] = 0


options = AnalyzeOptions(skip=1)
```

`fit_decay`、`fit_dual_decay`、`fit_rabi` 與 `fit_decay_fringe` 已回傳具名結果。
新程式碼讀屬性，tuple 解構只保留給既有 caller。以下的 `delays` 與 `real_signals`
是同長度的一維 float64 陣列，delays 非負，回傳時間沿用 delays 的單位：

```python
from zcu_tools.analysis.fitting import fit_decay


fit = fit_decay(delays, real_signals)
lifetime = fit.decay_time
uncertainty = fit.decay_time_err
curve = fit.curve
params, covariance, quality = fit.params, fit.cov, fit.quality
```

### Analysis 回傳值只放摘要

Analysis 回傳值的大小最多 O(log N)，N 是掃描點數或 shot 數。
可放擬合結果、純量、尺寸固定的小值，或每個 tone／qubit 一筆的結果。
隨掃描點數或 shot 數成長的陣列只在 analyze 內用來畫圖，不放進回傳值。
只發布圖的分析回傳 None。

GUI 在 primary、interactive 與 post-analysis 發布結果前呼叫 `summarize_analysis`。
任一 ndarray、list 或 tuple 攤平後超過 64 個元素會 raise ValueError。
訊息包含欄位路徑、元素數、上限與不能隨掃描大小成長的規則。
GUI 將違規顯示為該次分析的錯誤。存檔與 RPC 沿用同一個轉換函式。
64 是檢查上限，不代表 O(N) 的資料在少於 64 個元素時就合規。

### 來源選擇與分類 read axis

在 `make_form(ctx)` 用 `source(ctx, path, template)` 選來源。
`path` 接受 module slot 或 library root，可含 `$alias`。
Helper 查詢固定 capture 的 `ctx.has(path)`，有來源就回傳 `Slot(path)`。
這個 Slot 保持 reference，不 copy，也不附加 set。
缺 capture、缺 key、未設定或懸空的 slot、缺或未綁 alias 都讓 `ctx.has(path)` 回 False。
這時 helper 原樣回傳 Template，保留作者的 `set` 與 `seed_metadata`。
路徑語法錯誤、其他 alias 解析錯誤與來源故障直接傳出。
來源可解析但 family 不相容時，Form 仍報錯，不改用 Template。

Metadata 寫在 Template 上，Form 只有採用該範本才合併其 metadata。
Metadata key 使用 Form root-relative canonical path，同路徑以 `Form.with_seed_metadata` 的顯式值優先。
以下範例使用前述 core 的 `DelayCfg`，將選出的 node 放在 tree 的 readout 欄位：

```python
from zcu_tools.cfg_model import FieldMeta, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source

from .core import DelayCfg


def make_form(ctx: FormContext) -> Form[DelayCfg]:
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            seed_metadata={
                ("readout", "module", "gain"): FieldMeta(
                    description="Confirm readout gain before Run.",
                ),
            },
        ),
    )
    return Form(DelayCfg, tree={"readout": readout})
```

其餘必填欄位仍由實驗自己的 tree 提供。
若只宣告 `Slot(path)` 而不提供 Template，缺來源時 required 欄位保持 Invalid。
Nullable 且預設 None 的欄位保持停用且有效。

兩分支共用 `set` 時，在選出的 node 上做一次 `dataclasses.replace`。
以下指派放在 `make_form(ctx)` 內，再將 `pi_pulse` 放進 tree 的同名欄位。
若實驗的 Run 會覆寫 waveform.length，用下例隱藏它，並保留 node 原有的 set：

```python
from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Template

from zcu_lab.v2.support.measure.ctx_helpers import source


pi_pulse = source(
    ctx,
    "$qubit.module.x180",
    Template("qub_probe", bindings={"qubit": "$qubit"}, set={"gain": 1.0}),
)
pi_pulse = replace(pi_pulse, set={**pi_pulse.set, "waveform.length": Hidden(0.0)})
```

Slot 與 Template 都有 `set`，不需分別 replace，也不擴充 `source` 的參數。
`Slot.set` 的初值仍只套用到 copy，這段不將 reference 改為 copy。

單 tone 的分類宣告用 `Classify("readout")`，不用依 readout 類型手選 tone 0。
`shot_sums(prog, [axis])` 在 program 編譯後解析 label；pulse 或單 tone mux 自動選取。
多 tone mux 必須明寫原始 tone 序號，否則錯誤列出可選 tones：

```python
from zcu_tools.program.v2 import Classify


axis = Classify("readout")
mux_axis = Classify("readout", tone=2)
```

### 圖的建立、顯示與更新

建圖、預設樣式、呈現、owner-thread 更新與 finish／release 的契約見
[plotting README](../../lib/zcu_tools/plotting/README.md)。
Lab 的 live artists 與多面板格式見 [v2 README](README.md#即時圖工具)。

以下片段在 show 前填入資料，show 後修改標題。實驗自行選座標、labels 與圖名：

```python
from zcu_lab.v2.support import live_line

fig, ax = plots.subplots(num="measurement")
ax.set(xlabel="Delay (us)", ylabel="Signal")
line = live_line(ax)


@plots.on_plot_thread
def update(data):
    return line.set(delays, data)


update(real_signals)
fig.show()


@plots.on_plot_thread
def set_title(title):
    return (ax.set_title(title),)


set_title("Acquired signal")
```

## 使用者實驗的測試

個別 adapter、workflow 與 recipes 的測試邊界見
[程式碼品質](../../docs/code-quality.md#測試驗證可觀察契約)。
實驗定義改動依 [tests 的驗收規則](../../tests/README.md#實驗定義) 執行離線 probe；
最小腳本與命令見 [lab README](../README.md#實驗定義的離線驗收)。

## 實驗家族 records

RunRecord、explicit source 與具名圖的共同契約見
[experiment README](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。
各實驗的輸入、分析選項與結果型別由 leaf 的 core 定義，操作與 writeback 見同目錄 guide。
本頁不保存逐家族的遷移紀錄。

Mux 的 zip acquisition、participants、座標形狀與 partial data 見
[v2 README](README.md#mux-qubit-numeric-cores)。
操作入口見 [Mux two-tone spectrum](mux/twotone_spectrum/guide.md) 與
[Mux Amp Rabi](mux/amp_rabi/guide.md)。

Readout power 的八分支 QND 序列、native tag、指標與選擇 gain 的流程見
[Power guide](twotone/ro_optimize/power/guide.md)。

## 目錄佈局

每個 leaf 的位置與附件見[套件 README](README.md)。量測核心不 import GUI。跨實驗的 singleshot numerical helpers 留在 `support/singleshot/`，reset fit 與 JPA optimizer 留在各自的 leaf，通用 runtime 與量化工具留在框架。


---

## 實驗特有的分析與保存

實驗基底、RunRecord 與 Result 契約見[父層實驗介面](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。以下記錄 v2 實驗特有的分析與資料形狀。

有分析選項的實驗以 `Experiment.Options` 公開其既有 options 類別，例如 `T1Exp.Options(skip=1)`。GE 另以 `GE_Exp.PostOptions` 提供 post-analysis 選項。類屬性引用型別，不保存共用 instance；欄位與預設仍由原 options 定義擁有。沒有選項的實驗不補空 Options。Experiment 基底不要求這些屬性。

CKP numeric analysis先從ground/excited maps抽取resonance trace，再透過
`analysis.fitting.shared`共同擬合Lorentzian baseline、scale與width；兩個resonance
frequency維持local，兩個slope各自固定為零。Chi/kappa uncertainty直接由named global
covariance投影，包含local frequency cross-covariance；analysis不重建per-trace covariance
blocks。Backend minimum或covariance無效時fast-fail；`CKPAnalysis`只含chi、kappa與res_freq，圖交本次Plots。

`twotone/reset/RabiCheckExp` 以 explicit RunContext 執行，回傳純 Result，發布三分支 measurement。Analyze 接 RunRecord 與 None options，回傳純 `RabiCheckFit`，另向本次 Plots 發布 fit；接受 cfg=None。reset前分支決定
共同IQ投影與基頻，reset-only以同基頻擬合，reset後追加Rabi的分支包含基頻與二次諧波。
半峰對峰振幅、相位、相對contrast與殘差只描述平均IQ資料，不推導reset fidelity。
數值分析由同目錄`fit.py`擁有，GUI adapter只投影其純量；acquisition與持久化
資料格式維持三分支gain sweep。

`singleshot/ResetCheckExp` 以硬體 gain／branch sweep 保存 `ResetCheckResult.counts`，
shape 為 `(Ngain, 3, 3)`，順序是 gain、reset branch、G/E/Other。
三個 branch 依序是 reset 前、reset-only、reset 後追加 Rabi，屬於獨立 preparations，不是配對 shot 軌跡。
Counts 是已取得 rounds 的分類計數總和。Analyze 先按每點總計數轉成比例，
再選用 `ResetCheckAnalyzeOptions.confusion_matrix` 修正，回傳 reset-only 的平均／最差 excited population
與最大 Other population，不提供 IQ 校準 writeback。
Other 不是校準後 leakage，這些 population 不是 reset-channel fidelity。
Run 使用 explicit RunContext，回傳純 Result。Analyze 接 RunRecord 與 ResetCheckAnalyzeOptions，回傳純 ResetCheckAnalysis，另向本次 Plots 發布 `populations` 圖。

`singleshot/AmpRabiExp`保留硬體gain sweep，每個host round擷取所有gain的Reps筆raw IQ；
不同round的shots串接而不平均，canonical complex128 shape為`(Ngain, Reps * Rounds)`。
`shot_indices`是inner axis，`gains`是outer axis。完成的round立即更新live分類population；
取消時未完成round保持NaN，analysis僅移除完全缺失的shot columns，部分缺失的sweep明確拒絕。
Run 使用 saved readout 的 circle／radius 作 live 分類，缺 classifier 時省略 preview。
Raw acquisition 與 joint fit 不要求先有校準，analysis 不使用外部 confusion matrix。

Amp／Len Rabi、Check、ResetCheck 與 AC Stark 使用 explicit RunContext 執行，GUI 將同次 cfg／Result 配成 RunRecord。同步 analyze 接 source／typed options／Plots，不保存跨次成功狀態。
Check 在 Analyze 才要求 saved cfg 的 circle／radius；其他分析保留各自的 nullable cfg 契約。Amp 回傳 AmpRabiFit，Len 回傳 RabiJointFitSummary，兩者的 readout 診斷只讀 source.cfg。Check 只發布分類 scatter，無數值結果。AC Stark 回傳 AcStarkAnalysis，另提供 explicit source／AcStarkPlotOptions 的 population 圖入口；即時 Ground／Excited／Other 熱圖及 current trace 各自具名。

Amp 與 Len Rabi 共用 `zcu_tools.analysis.fitting.singleshot_rabi` 的 raw-IQ joint likelihood。
各實驗在自己的 analyze 畫 population、histogram 與 confusion matrix 診斷圖。
Amp固定無衰減／零相位，Initial State描述zero-gain前的主要狀態，支援signed gain。
Amp額外由joint frequency及其covariance推導pi/pi2 gain與誤差，保留原有scalar summary。
兩者的GUI校準writeback共用backend validity及finite校準值檢查。
舊Amp population-only檔案缺少IQ shots，無法轉換成新資料，也不在load時虛構raw IQ。

---

Singleshot T1／T1WithTone／T1WithToneSweep 使用 explicit RunContext，uniform 歸 typed cfg。同步分析接 RunRecord、typed options 與本次 Plots，允許 cfg=None。Tone 回傳純 t1／t1_b，其餘兩者只發布 fit 圖。即時 ground／excited traces 各自具名；Tone sweep 的六張 population 熱圖與兩張 current trace 各自具名。Transition-rate fitting、host rounds 與完整 outer-row 篩選保持不變。

### 版本實驗的資料形狀

`Experiment`、`AxesSpec`、inner-first 軸序及 grouped variables 的共同映射見[父層 README](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。以下保留 v2 實驗的具體資料形狀。

- **grouped experiment variables**：`CPMG_Exp` 使用 variables `lengths` / `signals`，axes 為 inner-first 的 `Time Index`、`Number of Pi`，盤上 `lengths` 單位為 seconds，記憶體內仍回復為 us。RO auto-optimize 使用 variables `readout_freq` / `readout_gain` / `readout_length` / `snr`；JPA auto-optimize 使用 variables `jpa_flux` / `jpa_freq` / `jpa_power` / `jpa_phase` / `snr`，其中 `jpa_flux` 以中性 device-native value 寫盤（unit `a.u.`、identity scale、數值不縮放），舊 auto grouped file 若 `jpa_flux` variable unit 為 `A` 不是 canonical，strict loader 不做 `A` fallback。頻率與時間在 disk 上使用 SI units（Hz、s），typed loader 重建回 Result 記憶體單位（MHz、us）；JPA phase 是 integer index。`load_run_with_metadata(path, spec=Exp.AXES_SPEC)` 讀取與 grouped spec 相符的 native `data.h5`，並透過 `restore_run` 還原 typed record。Labber 匯出由獨立的 `save_labber(record, path, spec=Exp.AXES_SPEC)` 提供，不能直接交給 native loader。既有 Labber 舊檔依 [migrations](../../migrations/README.md#legacy-resultdatabase--native-storage) 提供歷史證據並離線轉換；legacy `.npz` 或 sidecar 不是 runtime 可載入格式。
- **legacy single-file**：舊 Labber HDF5 的 `Frequency` `MHz/Hz`、`Yoko` flux 軸或 `ADC unit` signal channel 不符合當前 `AXES_SPEC`，不由 runtime/GUI 隱式轉換。`onetone/flux_dep` 的 canonical axes 是 `(freqs, values)`，對應 Result-native `signals.shape == (Nflux, Nfreq)`。
- **single-variable 離散狀態軸**：bath reset freq-gain 把四點 pi/2 tomography phase 視為同一個 Result 的第三個 sweep axis；bath reset length 把 phase 視為第二個 axis，Result-native shape 為 `(Nlength, 4)`；`CKP_Exp` 把 ground/excited prepared state 視為 `initial_states` axis；`GE_Exp` 把 ground/excited prepared state 視為 `prepared_states` axis，Result-native shape 為 `(2, Nshot)`；singleshot `len_rabi`以`shot_indices`作inner axis，canonical `complex128` raw-IQ shape為`(Nlength, Nshot)`。分析數學見 [analysis owner](../../lib/zcu_tools/analysis/README.md#singleshot-初態與-joint-fit)。舊 population-only 檔案缺少 IQ shots，canonical loader 明確拒絕而不虛構資料。
- **分類計數與狀態軸**：以下 Result 的 `counts` 保存完整 `[g,e,other]` 計數，`population_states=[0, 1, 2]` 是最內層軸。比例由 counts 正規化，analysis 不重建 other。

  | 實驗 | Result-native counts shape | 其餘軸，由外到內 |
  | --- | --- | --- |
  | MIST `power`／`freq`／`pre_freq` | `(Nsweep, 3)` | 對應的 gain／frequency |
  | Singleshot `ac_stark`、MIST `power_freq` | `(Ngain, Nfreq, 3)` | gains、freqs |
  | Singleshot `t1`／`t1_with_tone` | `(Nt, 2, 3)` | lengths、initial_states |
  | Singleshot `t1_with_tone_sweep` | `(Nx, 2, Nt, 3)` | xs、initial_states、lengths |

  舊兩欄 population 檔案不能直接交給新的 analyze，不提供轉換 adapter。
  Cfg 未變，`cfg_schema_version` 維持 `3.0`，不作 Result 資料版本。
  這類 homogeneous Result 的 native 資料存成單一 `data.h5`，Labber 另匯出單一 `.hdf5`。
  離散狀態不是 Data Variable，不拆成多個 sidecar artifact。Runtime 不載入 legacy artifact，
  也不重排舊 singleshot population HDF5 的 `(2, Nsweep)` 或 multi-sidecar z 方向。

Singleshot MIST 的 Freq／Power／PreFreq／FreqPower 使用 explicit RunContext，回傳純 Result。所有 classification 校準值屬於 typed cfg。同步 analyze 接 RunRecord、對應 AnalyzeOptions 與 Plots，允許 cfg=None，只發布 fit 圖。Power 支援 photon 軸與 log scale；FreqPower 不宣告未實作的軸選項。單 sweep 即時圖使用 measurement，二維掃描的 Ground／Excited／Other 熱圖各自具名；圖的 `Population` 標籤表示 counts 正規化後的三態比例。

Singleshot Len Rabi的length軸由host-side `Schedule.scan`逐點執行；每個program只擷取
單一length的`shots`筆raw IQ，再寫入sweep-first Result row，避免FPGA同時配置完整
`Nlength × Nshot` raw buffer。stop保留已完成rows，其餘維持partial-result NaN。

Joint-fit 數學與校準有效性見 [analysis owner](../../lib/zcu_tools/analysis/README.md#singleshot-初態與-joint-fit)。
Len Rabi 在自己的 core 畫三面板診斷圖，操作見
[leaf guide](singleshot/len_rabi/guide.md)。

Singleshot consumers 的分類參數由 RunRecord.cfg.readout 的 classifier 提供。
分類取數先 build program，再用 `shot_sums(prog, [Classify("readout")])` 解析讀出 label，交給 Schedule 的 `run_acquire_summed`。
Result 的 `counts` 保存各 round 的 class counts 總和，形狀與原有狀態軸相同；分析與 live plot 才按每點總計數轉成比例。
需要 Other 的讀出須有 circle 與 radius，不能用 linear 或 centroid3 表達 Other。
舊比例存檔缺少 counts 時明確拒絕，不以比例推算計數。
GUI confusion correction 只取 saved readout cfg 的 matrix；缺 cfg／classifier／matrix 時略過。
Rabi 與 Check 保留單 ADC raw IQ。Rabi 無 classifier 仍可 Run／fit，Check 在分析才要求 circle／radius。
Classifier 的複數持久格式由 cfg_model 與 core complex codec 擁有，lab 不另加 serializer。

檔案格式細節見 [datafile README](../../lib/zcu_tools/datafile/README.md)。

---

## 典型 `Exp.run()` 範本（以 `onetone/freq/core.py` 為例）

一般核心實驗的入口是 `run(cfg, *, context: RunContext) -> Result`。硬體與具名 devices 由 context 傳入，cfg 只含這次操作的設定。Result 只含資料。NotebookAdapter 或 GUI adapter 將原始 cfg 與 Result 配成 RunRecord，不在核心快取上一輪結果。

1. 以 `sweep2array` 將 sweep 展成硬體格點。Device setup 使用 `context.devices`。
2. 用 `fig, ax = context.plots.subplots(num="measurement")` 建圖並設定 labels。從 `zcu_lab.v2.support` import `live_line`，以 `line = live_line(ax)` 掛上曲線。呼叫 `fig.show()` 顯示圖，不使用 pyplot current state。
3. 以 `@context.plots.on_plot_thread` 裝飾 update 函式，本體回傳 `line.set(xs, signals)`。`SignalBuffer` 的 on_update 將完整 buffer 交給 update。Schedule 不持有 cfg，透過 `run_acquire*` 取數並寫回 buffer。
4. Schedule 接 `scope=context.scope`，將外部取消傳給 acquire。Experiment setup helper 接 `cancel_signal=context.scope.stop`，只在 helper 到 driver 的邊界轉成 Event；直接 acquire 的路徑也必須傳入 cancel flag。
5. 回傳純 Result。同步分析接 `analyze(source, options, *, plots)`，數值直接回傳，圖向本次 Plots 發布。

Schedule 的 buffer、retry 與 partial-result 規則見 [runtime README](../../lib/zcu_tools/experiment/v2/runtime/README.md)。
目前所有 Experiment 均走新 runtime 或直接 `SignalBuffer` path，包含 `lookback/core.py`、`onetone/*`、`twotone/*`、`singleshot/*`、`jpa/*`、`fastflux/*` 與 `mist/*`。

`onetone/freq` 的 frequency sampling 有兩種 mode：`linear` 沿用 program-side
`SweepCfg` sweep；`homophasal` 保留同一個 `freq` 使用者介面，但由已擬合的 resonator
circle 參數產生非等距 frequency array，再交給 `builder.add_table_sweep("freq", freqs)`。
兩種 mode 都用 `PulseReadout("readout", cfg.readout, freq=freq)` 播放並取數。
QICK 逐點轉成 generator/readout frequency words，原生 loop 收尾在每發後載入下一點。
最後一點回繞第一點供下一個 outer repetition 使用，不重置 DDS phase accumulator。
非等距模式仍先把點 round 到硬體格點，並在相鄰點 collapse 時 fast-fail。
存檔的 `freqs` 軸與 cfg 不變。

`zcu_tools.experiment.v2.utils.t1_sampling` 擁有 T1 family 的 non-uniform sampling contract，供
`twotone/time_domain/t1` 與 singleshot `t1`、`t1_with_tone`、`t1_with_tone_sweep` 共用：
在硬體量化前沿 normalized T1 decay curve 等弧長配置 delay，保留 configured window 與 point
count。各路徑依 delay 或 probe-pulse 所屬的硬體 grid 量化；任何 collision 或非嚴格遞增都
fast-fail，不以 sorting、`unique` 或自動減點掩蓋。Direct delay list 走相同 conversion
contract，但不重新採樣。Singleshot `t1_with_tone` 的 zero-length 點略過 probe pulse，避免
建立硬體不接受的 zero-cycle pulse；`t1_with_tone_sweep` 則要求完整 time axis 嚴格大於
零，並在 device setup 前 fast-fail。其 uniform 與 non-uniform time axis 都由
`TableLengthPulse` 在板上執行：外層 gain/frequency 維持 host scan，內層 length loop 從
dmem 載入 pulse length 與實際 duration；const/flat-top 共用單一 wmem template，readout
依每點真實 pulse 結束時間對齊，不把短 pulse 補到最長點。

### Sweep 參數與 cfg 的歸屬

以 module constructor keyword 傳入掃描值，不修改 caller cfg。
均勻掃描用 `builder.add_sweep(name, spec)`，任意物理值序列用 `builder.add_table_sweep(name, values)`。
兩者都建立頂層 loop，依宣告順序由外到內排列，重名會 raise ValueError。
查表回傳 `TableParam`，單位由參數決定，例如 freq 用 MHz、delay 與 length 用 us。
空序列或非有限值會 raise ValueError，不把不等距點轉成 `SweepCfg`。

```python
from zcu_tools.program.v2 import DelayAuto, PulseReadout

freq = builder.add_table_sweep("freq", freqs_mhz)
delay = builder.add_table_sweep("delay", delays_us)
builder.emit(DelayAuto("gap", t=delay))
builder.emit(PulseReadout("readout", config.readout, freq=freq))
```

範例使用尚未 build 的 `builder` 與非空序列 `freqs_mhz`、`delays_us`，frequency loop 在外，delay loop 在內。
T1 decay delay 取樣也將 `table.times_us` 交給 `add_table_sweep`，再用 `DelayAuto(..., t=delay)`。
需要同一 loop 的另一張表時，從 `zcu_tools.program.v2` import `TableParam`，以 `TableParam(name, values)` 建立。
表長與 loop 次數不同會在 build 時 raise ValueError。
`TableParam` 可與數字或未掃描的 `QickParam` 做 `+ - * /`，包含反向運算，結果仍是同一 loop 的新表。
與任何另一個掃描參數運算都會 raise TypeError，包含同 loop 的另一張表。

所屬 loop 每圈結尾都載入下一點，即使該發沒有播放該 pulse。最後一點後回到第一點。
Const 與 flat-top 的 length 可查表，flat-top 只改 flat 段，其他 waveform 的 length sweep 限制不變。
`MuxReadout` 不支援查表掃描，`ro_length` 的拒絕時機見[constructor 覆寫](#cfg-名稱順序與取數欄位)。
時間 getter 只在量化後逐點值可表示成常數或仿射時回傳等值 `QickParam`，不回傳 `TableParam`。
非仿射或 runtime-only 時間會 raise RuntimeError。
具名 timeline duration 逐點 ceil，原生 QICK 轉換保留 nearest，不能假設兩種入口的取整相同。

查表每點的硬體字嚴格等於該值的純量 program。
線性掃描的實際浮點值回編到 gain 等截斷取整欄位時，可能與線性字差 1 個最小單位，因為浮點回編不保證冪等。
要比較 linear/table parity，先以同一 profile 確認每個實際點的純量字等於線性字。
查表與線性值的轉換入口見 [program 查表掃描](../../lib/zcu_tools/program/v2/README.md#查表掃描)。

Schedule 與 step 不接收 cfg；需要修改設定時，作者在 run 內明確 deepcopy。
`sweep2array` 只讀 sweep 定義與通道，座標建立不依賴 pulse param mutation。

以下操作直接使用作者的 run inputs：

- **device setup**：`set_*_in_dev_cfg(cfg.dev, ...)` + `setup_devices(cfg, context.devices, progress=True, cancel_signal=context.scope.stop)` 需要在掃描前先把硬體帶進度地初始化到起點，留在 `run()` body；`progress=True` 本身即通知，不另加 warn。
- **Raw-shot program 設定**：`singleshot/ge`、`singleshot/check`、`singleshot/len_rabi` 與 `mux/singleshot_ge` 的實驗 cfg 不含 reps／rounds。`run()` 直接將 `reps=cfg.shots` 與 `relax_delay=cfg.relax_delay` 交給 builder，acquire 明確使用 `rounds=1`。有宣告 initial_delay 的實驗同樣明傳。Lookback 固定 `reps=1`，`run_acquire_decimated(prog, context.soc, rounds=cfg.rounds)` 沿用可編輯的軟體平均次數。這些 runtime 固定值不出現在 Form。

---

## Timeline program 與 Schedule 取數

`TimelineBuilder` 只記錄 program 動作，`build()` 產生並編譯 `TimelineProgramV2`。
它不接收 soc，不取數，也不是 context manager。
需要自訂 register 運算或 dmem 操作時，改寫 `TimelineProgramV2` 子類別。
兩種寫法都把 program 交給 Schedule 或 scan step 的 `run_acquire*`。

SimulationSoc 從 compiled stream 建立 acquisition request。
編譯與 IR timing analysis 不需要取數；離線取數須使用對應的 simulation environment。
支援邊界見 [program simulation](../../lib/zcu_tools/program/v2/simulation/README.md)。

`delay_by_timeline` 依每個掃描點的 runtime 結尾推進 reference，shell 收尾 WAIT 使用編譯期常數。
已知合法掃描點間的剩餘時間差距大於 0 時編譯發 warning，超過 `relax_delay` 時 raise ValueError。
未知差距只警告上界，不以該上界拒絕編譯。
兩種寫法共用 [ADR-0083 的收尾 WAIT 契約](../../docs/adr/0083-program-macros-and-timelines.md#收尾-wait-與-slack-護欄)。
Scheduled timing analysis 不驗證 CPU 等待與跨 shot 實機排程。

### 用 TimelineBuilder 寫序列

以下片段放在實驗 `run(config, *, context)` 中。
`config` 提供 `length: SweepCfg`、`pi_pulse: Pulse.Cfg`、`readout: Readout.Cfg`，
以及 `reps`、`rounds`、`relax_delay` 與 `earlystop_snr`。
`on_update` 是本次實驗的 live plot callback，接收完整 signal array。
直接讀取傳入的 `config`。Schedule 與 step 只保存控制流程、env 與 buffer path。
需要修改 pulse cfg 時，在 `run()` 開頭自行 deepcopy，不修改 caller 的 config。

```python
import numpy as np
from zcu_tools.experiment.v2.runtime import RunOptions, Schedule, SignalBuffer
from zcu_tools.experiment.v2.utils.snr import snr_checker
from zcu_tools.program.v2 import Pulse, Readout, TimelineBuilder

signals = SignalBuffer((config.length.expts,), on_update=on_update)
with Schedule(signals, scope=context.scope) as sched:
    builder = TimelineBuilder(
        context.soccfg, reps=config.reps, relax_delay=config.relax_delay,
    )
    delay = builder.add_sweep("length", config.length)

    drive = Pulse("pi_pulse", config.pi_pulse)
    readout = Readout("readout", config.readout)
    builder.emit(drive)
    builder.timeline_inc(delay)
    builder.emit(readout, tag="terminal")

    prog = builder.build()
    sched.run_acquire(
        prog, context.soc, rounds=config.rounds,
        options=RunOptions(
            stop_condition=snr_checker(signals.at(), config.earlystop_snr, np.abs),
        ),
    )
```

`add_sweep` 同時宣告 loop 並回傳 `QickParam`，所以獨立寫一行，不藏在 module 建構子中。
多個 loop 依宣告順序由外到內。非均勻物理值用 `add_table_sweep(name, values)`，
用法與限制見 [Sweep 參數與 cfg 的歸屬](#sweep-參數與-cfg-的歸屬)。
`add_loop(name, count)` 加 `LoadValue` 保留給整數 selector 或候選 pulse 選擇。
Builder 的 `emit`、`delay`、`delay_auto`、`timeline_inc`、
`delay_by_timeline`、`pulse`、`trigger`、`fork`、`join`、`timeline`、`repeat` 與 `branch`
沿用 program 層的同名 API。原生布林控制選項用 keyword，例如
`builder.delay_auto(0.1, gens=False, ros=True)`。
`build()` 後 builder 凍結，再記錄動作或再次 build 都會 raise ValueError。
平行發射用 `fork` 建立各 child timeline，依作者順序 `emit`，最後 `join` 合併結尾。
TwoPulseReset／BathReset 的 child 與 branch case 不得推進 reference。
原生 `repeat` 與 `branch` 直接寫在 builder body。
`join` 只合併 timeline 結尾，不發射序列或改變 reference。

### 手寫 TimelineProgramV2

手寫版本在 `_initialize` 建立 module 並 `declare`，在 `_body` 發射。
下例與上例使用相同的輸入欄位，每次建構都複製 cfg 並建立新的 module。

```python
from copy import deepcopy

from zcu_tools.program.v2 import Pulse, Readout, TimelineProgramV2


class T1Program(TimelineProgramV2):
    def __init__(self, soccfg, cfg):
        self.inputs = deepcopy(cfg)
        super().__init__(
            soccfg, reps=cfg.reps, relax_delay=cfg.relax_delay,
        )

    def _initialize(self, _cfg):
        cfg = self.inputs
        self.delay_param = self.add_sweep("length", cfg.length)

        self.drive = Pulse("pi_pulse", cfg.pi_pulse)
        self.readout = Readout("readout", cfg.readout)
        self.declare(self.drive)
        self.declare(self.readout)

    def _body(self, _cfg):
        self.emit(self.drive)
        self.timeline_inc(self.delay_param)
        self.emit(self.readout, tag="terminal")


signals = SignalBuffer((config.length.expts,), on_update=on_update)
with Schedule(signals, scope=context.scope) as sched:
    prog = T1Program(context.soccfg, config)

    sched.run_acquire(
        prog, context.soc, rounds=config.rounds,
    )
```

同一 module 物件可以傳給多個 `TimelineBuilder` 或原生 `TimelineProgramV2`。
每個 program 依 module 物件身分保存宣告結果，重新 compile 時配置自己的資源。
錄製或編譯後不要修改 module 或它的 cfg。不同掃描設定仍要建立不同 module。
同一 program 內可以多次 emit 同一物件，宣告只做一次，每次 emission 的時間與 tag 各自獨立。
重複 readout 用不同 tag 區分，例如 `"herald"` 與 `"terminal"`。
相同 ADC 或 register 不能由兩個不同物件重複占用，衝突會在編譯時指出兩個物件。
Pulse 家族原有的 generator 與 template 共用規則不變。

### Timeline 範圍只切換預設名字

Timeline、reference、`timeline_inc`、`delay_by_timeline` 與 branch／repeat 的公開入口見
[program API 速查](../../lib/zcu_tools/program/v2/README.md#原生寫法與具名-timeline)。
範圍切換與控制流契約由 [ADR-0083](../../docs/adr/0083-program-macros-and-timelines.md#timeline-只存結尾與-reference-分開) 擁有。

### 用 run_acquire* 寫入與停止

Schedule 的 `run_acquire*`、slot、converter、stop 與 retry 契約見
[runtime README](../../lib/zcu_tools/experiment/v2/runtime/README.md#schedule-run-acquisition)。
以下只示範實驗如何接上 summed 與 decimated 取數。

`run_acquire_shots` 需要已進入 context 的 Schedule，但不需要預建 buffer，也不做轉換或 live update。
它只取一輪，`rounds` 省略或設為整數 1，其他值會拋出 `ValueError`。
需要多輪 raw shots 時，由作者在 host loop 重複呼叫。
它接受 `RunOptions(retry=..., progress=..., on_stop=...)`。
`converter` 或 `stop_condition` 非 `None` 時，入口會拋出 `ValueError`。
成功取數後，`program.get_raw()` 若回傳 `None`，入口會拋出 `RuntimeError`。
外部取消時入口回傳 `None`，不讀 raw buffer。
只取消 `"acquire"` 子 scope 不改父 Schedule outcome，下一次 acquire 可繼續。
只有父 Schedule 的 stop token 已設時，Schedule outcome 才記為 `stopped`。
Producer 失敗沿用上述 retry 預算，不做 slot 還原。Raw-buffer 讀取失敗不 retry。

Summed 範例先 build 再解析 read label。以下是上方 builder 範例的替代取數段，
使用同一個 `prog`，但 slot 改為 float64 的 count、sum I、sum Q 三欄。

```python
from zcu_tools.program.v2 import IQ, shot_sums

processor = shot_sums(prog, [IQ("terminal")])
sums = SignalBuffer((config.length.expts, 3), dtype=np.float64)

with Schedule(sums, scope=context.scope) as sched:
    sched.run_acquire_summed(
        prog, context.soc, processor=processor, rounds=config.rounds,
    )
```

需要不同形狀或 conditional mean 時，用 `RunOptions(converter=...)` 明確轉換。
`shot_sums`、分類軸與 `split_sums` 見 [program acquisition](../../lib/zcu_tools/program/README.md#summed-acquisition)。
Mux 多 ADC 的 run 也要提供符合 slot 形狀的 converter，不能沿用單 ADC 的預設 converter。
`MuxReadout.collect_raw_iq(prog)` 在 completed-round hook 讀取該 program 的 raw buffers，不把累積 averaged IQ 當作逐 shot IQ。

Decimated 範例不使用 sweep。`trace_buffer` 由 caller 按這次 rounded readout window
預先配置，形狀須符合 backend 的 repetitions 與 samples 軸，並可提供 `on_update`。
範例建立新的 readout，也可以重用設定不變的 module；宣告結果由各 program 持有。

```python
trace_builder = TimelineBuilder(context.soccfg, reps=1)
trace_builder.emit(Readout("readout", config.readout))
trace_prog = trace_builder.build()

with Schedule(trace_buffer, scope=context.scope) as sched:
    sched.run_acquire_decimated(
        trace_prog, context.soc, rounds=config.rounds,
        options=RunOptions(progress=True),
    )
```

Program 本身的 `acquire`、`acquire_summed` 與 `acquire_decimated` 只取數並回傳。
它們不寫 Schedule slot，不管理 live update 或 Schedule stop_condition。
直接呼叫 `program.acquire*` 時，作者自行負責以下事項：

- 傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`，讓 program 收到 Schedule 的停止請求。
- 接住 `StoppedPartialAcquireError` 並決定是否呼叫 `sched.set_stop()` 或 `step.set_stop()`，以及如何處理未完成的資料。
- 直接呼叫不建立 `"acquire"` 子 scope，所以 `scope.cancel("acquire")` 停不到這次取數。
- 直接呼叫不會自動 retry，需要重試時由作者撰寫重試流程。

直接取數仍是作者的自由。Runner 入口是代管 scope、取消轉譯與 retry 的慣例工具，不限制 core 的取數方式。

## v2 cfg 組合慣例

實驗 cfg 從 `zcu_tools.cfg_model` import `ConfigBase`。Module 與掃描欄位直接放在 cfg 頂層，不宣告 Modules／Sweep group。Form 的排序見[寫法慣例](#cfg-名稱順序與取數欄位)，模組保留 selector 與子表單，掃描保留 `SweepCfg` 子表單。`FieldMeta` 仍可宣告 label、unit 等顯示提示。Pulse 的 Advanced group 保留 NQZ、Phase、Pre-delay、Post-delay 與 Mixer freq 的既有摺疊。Tone lists 保留逐元素結構。只宣告實驗使用的 reps、rounds 與 relax_delay。Initial delay 只有實驗需要編輯或驗證時才宣告，其餘沿用 program 的 1.0 us 預設。

TimelineBuilder 不讀 cfg。作者明傳 reps、relax_delay 與需要的 initial_delay。
`builder.build()` 編譯 program；取數時明傳 `rounds=config.rounds` 給 Schedule 的 `run_acquire*`。
Live helper 見[圖的建立、顯示與更新](#圖的建立顯示與更新)。

平鋪 cfg 使用 `cfg_schema_version="3.0"`。Load 不轉換舊巢狀 cfg。舊 major 的資料仍載入，record.cfg 為 None，cfg_issue 說明原因，資料陣列與 metadata 不變。

Core import 實際使用的 module，型別選擇見[寫法慣例](#cfg-名稱順序與取數欄位)。`SweepCfg` 與硬體指令仍按需 import。`SweepCfg` 定義在 `program/v2/sweep.py`。父層 cfg 型別與執行前組裝方式見[父層 README](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。

---

## Signal/Raw 處理慣例

- 除了用來產生分類參數的 `singleshot/ge`、`mux/singleshot_ge`、`singleshot/qutrit_population`、`singleshot/amp_rabi`、`singleshot/len_rabi`、`singleshot/check`，實驗不保存逐 shot 資料。分類在取數時做，Result 存計數或總和，避免為事後分類保存整批 IQ。校準實驗仍保留逐 shot 分布，供分類參數的擬合與驗證使用。
- **`raw` 格式**：QICK 回傳的 raw 一般是 `list[ndarray]`，第一個元素 shape 為 `(nro, ..., 2)`（IQ 兩個實數）。
- `default_raw2signal_fn` 定義在 `zcu_tools.experiment.v2.runtime.acquisition`。預設轉換的 ADC／read 限制與多 ADC 的 converter 要求見 [runtime owner](../../lib/zcu_tools/experiment/v2/runtime/README.md#schedule-run-acquisition)。
- **自訂 raw conversion**：integrated 與 decimated acquire 都以 `RunOptions(converter=...)` 傳入轉換。Converter 收各輪累積與最終 raw，回傳符合 slot shape 的 ndarray。
- Live plot 的 `signal2real` 可由實驗本地定義，或重用 lab support 的轉換工具。實驗選擇 magnitude、phase 或 real；analyze 可使用不同的轉換。
- **scalar/array 邊界**：座標轉換工具（例如 value↔flux）可接受 scalar 或 ndarray；若後續 plotting/analysis 需要 indexing、min/max 或與另一個 sweep array 對齊，呼叫端在邊界用 `np.asarray(..., dtype=...)` 正規化成 ndarray，而不是用型別宣告假設回傳一定是 array。
- **peak-picking smoothing**：ro-optimize 與 reset 這類以 SNR/map argmax 找最佳點的分析預設使用 `smooth_method="wavelet"`；`smooth_method="gaussian"` 保留為舊 Gaussian 對照。`smooth` 是通用強度：Gaussian 時是 sigma，wavelet 時是 threshold scale。ro-optimize length 的 `t0` 是 `SNR/sqrt(length + t0)` 的 duration-normalization term；`t0 > 0` 啟用短 readout bias，且較小的正值 bias 較強。`None` 與 `0.0` 都是純平滑 SNR argmax。

---

父層 `experiment/utils/` 的 comment 與 sweep helper 見[父層 README](../../lib/zcu_tools/experiment/README.md#cfg-與父層支援)。儀器角色 helper 位於 `support/device.py`，見 [lab v2 說明](README.md#實驗角色與-device-操作)。

## v2 工具與實驗輔助

- **`sweep2array(sweep_cfg, name, {"soccfg", "gen_ch", "ro_ch"})`** — 展開 `SweepCfg` 為 numpy array，已套 ZCU 量化（`round_zcu_freq/time/gain/phase`）。Exp 幾乎都靠這個產生 x 軸。
- **`round_zcu_*`** — 單點版本的量化函式；`round_sweep_dict` 同時處理整個 sweep dict。時間的量化有特殊處理：預先減去 `0.5 * one_cycle` 以匹配 QICK sweep 用 `np.trunc` 的行為。多點 sweep 的 step 若在量化後變成 0，會 fast-fail 並要求放大 span 或減少 expts，避免 GUI/agent 看到低階 `SweepCfg` 一致性錯誤。
- **`merge_result_list(list_of_results)`** — 把 `list[dict[name, ndarray]]` 遞迴轉成 `dict[name, ndarray]`（外層 list 變成最外層維度）。`ResultTree.measurement_result(name)` 只對單一 measurement 呼叫它，並快取 stacked view；更新某個 measurement 時不重算 unrelated measurement。
- **`estimate_snr` / `snr_as_signal` / `snr_checker`** — SNR 估計與 early-stop：`estimate_snr` 搭配 `snr_checker` 用於曲線 early-stop；`snr_as_signal(raw, ge_axis, skew_penalty=0.0)` 從 `MomentTracker` 的 g/e IQ moments 計算 pooled-sigma separation SNR。`skew_penalty=0.0` 是純 SNR；提高 `skew_penalty` 會以連續 rational penalty 降低 projected skew 與 g/e shape mismatch 較大的候選點，供 readout optimization 與 JPA optimization 共用。

---

## Lookback & Tracker

- `LookbackExp`（`lookback/core.py`）使用 `run(config, *, context)` 與 `analyze(source, options, *, plots)`。Result 只含 us time axis 與 complex signals，cfg 隨 RunRecord 保存。Run 沿 Schedule 的 decimated acquire，runner reps 強制為1，caller cfg 不改。LookbackAnalyzeOptions 保留 core ratio0.3／smoothNone，分析回傳只有 predict_offset 的 LookbackAnalysis，fit 圖寫入本次 Plots。分析從 magnitude 峰值向前找最後一個低於門檻的 sample，Gaussian smoothing 先作用於 complex signals。Cfg None 仍可分析，預設 saver 則拒絕；canonical Time 軸在磁碟上仍為 seconds。Notebook 使用共用 NotebookAdapter，沒有核心 last_result。
- **`KMeansTracker`**（`zcu_tools.experiment.v2.utils.tracker.kmeans`）：線上維護 `(..., 2)` IQ 樣本的動態多群統計（每群 `cluster_mean` / `cluster_covariance` / `cluster_center` / `cluster_weight`），支援 leading dims 與 `share_axis` 共用群組。內部以增量統計維護每群矩，無需保留原始樣本；singleshot 家族可直接用其 cluster 統計估計 SNR。

---

## Twotone RB 測量策略

`twotone/irb` 是獨立的 paired reference/interleaved 實驗，固定存
`(round, seed, arm, depth)` IQ；arm 0 是 reference，arm 1 是 interleaved。
GUI target 預設 X90，支援 X90、X180、Y90、Y180，沿用校準的 X90/X180 pulse。
兩組使用相同 random Clifford seed，target 在每個 random Clifford 後實際插入，
不合併相鄰 gate；各自計算完整 inverse，沿用連續 virtual-Z frame。
共享序列與 program 建構分別在 `twotone/rb/sequence.py` 與 `program.py`。
IRB 外層由軟體遍歷 round、seed、arm，每個 seed/arm 的 program 快取重用，
program 固定 rounds=1，depth/reps 在硬體完成。Arm 先後順序隨 round/seed 交替。
原 `twotone/rb` 的二維存檔與每個 program 內完成 rounds 的行為維持不變。

IRB analysis 只選完整 round/seed 配對，先平均同 seed 的 rounds，再等權平均 seeds。
兩組共用 IQ 投影軸，各自 fit `A*p**depth+B`；target error 為
`(1-p_interleaved/p_reference)/2`。95% interval 以整條配對 seed 曲線 bootstrap，
不包含 gate-dependent/coherent noise 的系統性誤差。負 error 不 clamp。
中止時未完成的 slot 維持 NaN，原始資料保存；不足兩個 paired seeds 或四個
distinct depths 時分析明確失敗。

- RB 量測採「每個 seed 一個 program」的結構：在 program 內用 sweep index 遍歷 depth，而不是在 host 端對 `(seed, depth)` 重複建立程式。
- recovery gate 是累積 Clifford 的**完整 group inverse**（24 種），不是只把態送回 +Z 的 state-restoring gate：module load 時從 `CLIFFORD_GROUP` 的 6-state permutation（對 24 元 quotient group faithful）程式化生成 `CAYLEY` / `INVERSE_INDEX` 查表，順序約定 `CAYLEY[i][j]` = 先作用 C_j 再 C_i、累積寫 `acc = CAYLEY[next][acc]`，兩處註解互相錨定，改其中一邊必須同步。
- seed 對應的 random gate prefix 長度與 recovery gate id 以 `LoadValue` 從 dmem 查表；random 段用 program 的 register-driven `repeat()`；recovery 因 inverse decomposition 最多含 2 個 physical pulse，使用兩個獨立 `ComputedPulse` slot（`recovery_gate_0/1`），不足處以 `BasicGate.Id`（gain=0）補位。兩個 slot 必須共用完整 `gate_pulses` candidate list，使 recovery 段時長與 depth 無關。
- random 段 `repeat()` 固定使用 `n="rand_len"`；當 random 序列可為空時，交由 `LoadValue(values=[])` 的 no-op 行為吸收，不在 RB 層做 sentinel value 邏輯。
- rounds 交給 `acquire()` 原生流程處理，host 層只掃 seed。

---

## 外部中斷支援（cancel_flag）

Schedule 的 acquire-local cancel flag、部分資料、retry 與 `on_stop` 契約見
[runtime 的 Stop 與 Retry](../../lib/zcu_tools/experiment/v2/runtime/README.md#stop-與-retry)。
直接呼叫 program acquire 時，作者明傳 cancel flag 並決定是否回傳已取得的資料。

## 寫新 Experiment 時的檢查清單

1. 定義 `XxxCfg(ConfigBase)`，頂層欄位與 module 型別依[寫法慣例](#cfg-名稱順序與取數欄位)宣告。平鋪、版本與 Advanced group 見[v2 cfg 組合慣例](#v2-cfg-組合慣例)。
2. 繼承 `Experiment[T_Result, XxxCfg]`，需要 typed record 持久化時宣告 class-level `AXES_SPEC`。Single 與 grouped mapping 共用此入口。Result dataclass 只放資料，cfg 與歷史 metadata 由 RunRecord 持有。以 `restore_run(stored, spec=Exp.AXES_SPEC)` 還原已讀 native Container，以 `load_run_with_metadata(path, spec=...)` 讀檔。`save_run(record, path, spec=..., replace=False)` 寫 native，`save_labber(record, path, spec=...)` 另作 Labber 匯出，兩者不猜格式。基底只宣告 Run 與分析簽名，不提供存讀方法。GUI 的 tag、restore 與保存內容 hooks 預設使用同一宣告，見 [GUI contracts](gui-contracts.md#共用-lifecycle-hooks)。
3. 實作 `run(cfg, *, context)`，沿用上方的 RunContext／Plots／Schedule 流程。同步分析另實作 `analyze(source, options, *, plots)`，互動分析由前端 helper 擁有。
4. Schedule 與 step 的 `run_acquire`、`run_acquire_summed`、`run_acquire_decimated`、`run_acquire_shots` 自動注入 acquire-local `cancel_flag`。
   `run_acquire_shots` 不寫 slot，外部取消時回傳 `None`。只有父 Schedule 的 stop token 已設才記 `stopped`。
   直接呼叫 `program.acquire*` 時，依[作者責任清單](#用-run_acquire-寫入與停止)傳入 `cancel_flag=sched.stop` 或 `cancel_flag=step.stop`，自行處理取消轉譯、scope 與 retry。
   SNR early stop 使用三個 slot 入口的 `RunOptions(stop_condition=snr_checker(...))`，不適用於 raw-shot 入口。
5. 持久化由 `AXES_SPEC` 宣告。軸用 `AxesSpec.Axis`，signal 用 `AxesSpec.Z`。MHz 轉 Hz 寫 `scale=1e6`，us 轉 s 寫 `scale=1e-6`；原值保存時省略 scale，預設為 1.0。盤上是 SI 單位，記憶體內維持習慣單位。`AXES_SPEC.tag` 取有層次的 on-disk 名字，例如 `"twotone/rabi/len"`，axes 以 inner-first 排列。Native 保存與還原依同一 spec 換算；`save` 的 Labber 匯出不是 `load` 的逆操作。Typed cfg 不相容時 native restore 保留有效 data、回 cfg=None 與 cfg_issue，缺 cfg envelope 或壞資料仍拒絕。來源與失敗契約見 [experiment owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。
6. 如果有多個 sweep 軸，先區分 host loop 與 program loop：host loop 用 `sched.scan(...)` / `sched.repeat(...)`，program loop 用 `TimelineBuilder.add_sweep(...)` 或 `add_loop(...)`。Host soft sweep 若需重用每點 program，在 `run()` 裡維護 dict；cache miss 時建立新的 builder 並 build，每次量測時呼叫 `step.run_acquire(program, context.soc, ...)`。

Singleshot GE/Rabi 的 radius 自動搜尋上限為 g/e 中心距離，手動分類可使用更大 radius。Raw IQ 投影 likelihood 不依賴 radius；derived confusion 與 population 使用圓內且最近中心的互斥區域。GE `consider_other` 僅控制 radius 選擇的 midpoint-other 懲罰模型，回傳校正矩陣維持 isolated-other row。

### Singleshot initial state

GE、Len Rabi 與 Amp Rabi 的 Initial State 只影響分析。操作、phase offset 與校準
writeback 分別見 [GE](singleshot/ge/guide.md)、[Len Rabi](singleshot/len_rabi/guide.md)
及 [Amp Rabi](singleshot/amp_rabi/guide.md)。
初態、零座標外推與 joint likelihood 的數學見
[analysis owner](../../lib/zcu_tools/analysis/README.md#singleshot-初態與-joint-fit)。

一般 Len Rabi 的 phase 選項見 [leaf guide](twotone/rabi/len_rabi/guide.md)，
Ramsey／Echo 的 fringe phase 選項見
[Ramsey](twotone/time_domain/t2ramsey/guide.md) 與 [Echo](twotone/time_domain/t2echo/guide.md)。
