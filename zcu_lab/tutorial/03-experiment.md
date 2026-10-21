# 新增實驗與 GUI 表單

這篇把 [twotone/power_dep](../v2/twotone/power_dep/) 複製成 `twotone/power_map`。
新實驗仍量測 qubit drive gain 對頻譜的影響，先用不同的名稱完成註冊，再改量測內容。
你會改五個 leaf 檔案，最後把 adapter 加進 [definitions.py](../definitions.py)。
不需要讀框架原始碼。各檔案的責任見 [adapter 目錄佈局](../v2/measure-authoring.md#目錄佈局)。

## 1. 複製 leaf package

在 repo root 執行：

```bash
cp -R zcu_lab/v2/twotone/power_dep zcu_lab/v2/twotone/power_map
```

下文的檔名都指新目錄。保留原目錄，原本的 `twotone/power_dep` 會繼續使用原 adapter。
先檢查以下清單，再逐檔修改。型別改名時，同步修改 import、型別參數與建構呼叫。

| 檔案 | 要改的位置 | 完成後的狀態與契約 |
| --- | --- | --- |
| `core.py` | cfg、Result、experiment 類別、`AXES_SPEC`、`run`，以及是否需要 `analyze` | 核心與持久化名稱獨立，見 [新 experiment 檢查清單](../v2/experiment-authoring.md#寫新-experiment-時的檢查清單)。 |
| `form.py` | `PowerCfg` import、Form 型別、初值樹、角色引用、Template／Slot | Open／Reset 能建立新 cfg 的輸入樹，見 [Form 與固定來源](../v2/measure-authoring.md#目錄佈局)。 |
| `gui.py` | core import、RunRecord alias、adapter 類別、capabilities、writeback | GUI 能建立核心並配對同次 cfg／Result，見 [GUI ownership](../v2/gui-contracts.md#ownership)。 |
| `guide.md` | 量測目的、必要校準、操作、結果判讀、writeback | operator 能知道何時能 Run，見 [operator guide 約束](../v2/measure-authoring.md#operator-workflow-guide-約束)。 |
| `__init__.py` | package 說明，若有 exports 也要更新 | import package 不啟動 GUI 或量測，見 [實驗目錄佈局](../v2/experiment-authoring.md#目錄佈局)。 |

## 2. 在 core.py 定義資料與量測

把 `PowerResult`、`PowerSweepCfg`、`PowerModuleCfg`、`PowerCfg`、`PowerExp`
依序改名為 `PowerMapResult`、`PowerMapSweepCfg`、`PowerMapModuleCfg`、`PowerMapCfg`、`PowerMapExp`。
保留 cfg 的型別關係。Module cfg 描述 pulse／readout，sweep cfg 描述 gain／frequency。
組合方式見 [v2 cfg 慣例](../v2/experiment-authoring.md#v2-cfg-組合慣例)。

這次不改資料形狀。`PowerMapResult` 保留 `gains`、以 MHz 表示的 `freqs`，
以及 shape 為 `(Ngain, Nfreq)` 的 complex IQ `signals`。
把 `AXES_SPEC.tag` 改成 `"twotone/power_map"`，並讓 `result_type`、`cfg_type` 指向新型別。
Axes 保持 inner-first 的 `freqs`、`gains`，頻率在檔案中以 Hz 保存。
之後若改 Result 欄位、軸序或單位，也要一起改 spec，見 [資料形狀](../v2/experiment-authoring.md#版本實驗的資料形狀)。

`run(cfg, *, context)` 先保留原序列。它對每個 host gain 掃描 FPGA frequency，
依序執行 reset、選用的 init pulse、qubit probe 與 readout，最後回傳純 Result。
Gain 是 drive amplitude，不是 dBm。之後改量測時，直接在 `run` 寫 pulse 序列與階段註解，
不要把整段序列藏進 helper，見 [core 中的序列與分析政策](../v2/experiment-authoring.md#1-在-core-直接寫出-pulse-序列與分析政策)。

原範本沒有實作 `analyze`，本篇也不新增 fit。
若你的實驗需要分析，在 core 實作同步 `analyze(source, options, *, plots)`，
從 explicit RunRecord 讀資料，回傳數值結果並向本次 Plots 發圖。
不要從目前表單補歷史 cfg，見 [分析與保存](../v2/experiment-authoring.md#實驗特有的分析與保存)。
這項變更也需要下一節的 FIT 宣告與 adapter hook，不能只加 core method。

## 3. 在 form.py 設定初值樹

把 `.core` 的 import、回傳 annotation 與 `Form(...)` 的 cfg 換成 `PowerMapCfg`。
這次把 gain 初值改為 `Range.between(0.001, 0.2, expts=31)`，
frequency 點數改為 `101`。這些只是新表單的起始值，不代表硬體安全範圍。

逐項核對原本的初值樹：

- `"$qubit.freq"` 與 `"1.5 * $qubit.kappa"` 決定 frequency centre／span。
  它們讀取 Open／Reset 的固定來源。缺校準時欄位保持 Invalid，不填猜測數字。
- `Template("qub_probe", bindings={"qubit": "$qubit"}, ...)` 產生 inline probe。
  保留 freq／gain 的 `Hidden(0.0)`，因為 `run` 會用 sweep 座標覆寫它們。
- `Slot("$qubit.module.readout")` 使用已選 library module。
  來源不存在時，原範本改用 `Template("readout", bindings={"resonator": "$resonator"})`。
- `Slot("$qubit.module.reset")` 是選用的 reset 來源。`init_pulse` 繼續隱藏。
  不要為了開表單而建立假的 library reference。

新增欄位時，先在 core cfg 定義型別與 FieldMeta，再在 tree 指定初值或顯示方式。
Tree 不另定義一份 cfg shape。Expression、Slot／Template 與 Reset 的契約見
[Form 與固定來源](../v2/measure-authoring.md#目錄佈局)。

## 4. 在 gui.py 接上 adapter 與 writeback

先把原本指向 `zcu_lab.v2.twotone.power_dep.core` 的絕對 import 改成：

```python
from .core import PowerMapCfg, PowerMapExp, PowerMapResult
```

把 `PowerDepRunResult` 改成 `PowerMapRunResult`，其中的 `RunRecord` 型別參數改用新 cfg／Result。
把 `PowerDepAdapter` 改成 `PowerMapAdapter`。
同步更新 BaseAdapter 型別參數、`exp_cls`、`experiment_tag`、`make_form`、
`run` 的型別與 `PowerMapExp().run(...)`。

保留 `RunRecord(cfg=cfg, result=result)`，讓 Save／Load 對應同一次量測。
`guide_path` 保持指向同目錄 `guide.md`，不指回舊實驗。
本篇保留 `analysis=AnalysisMode.NONE, load_data=True`，不加 analyze 或 writeback hooks。

若你新增 core fit，把 capability 改成 FIT，並在 adapter 提供 `analyze`。
需要寫回時，由 adapter 的 `get_writeback_items()` 提出具體 target 與值，
讓 operator 審閱後套用。沒有 fit 的 power map 不提出校準。
Hook 與 capability 必須一致，import 時就會檢查，見
[capability 宣告](../v2/gui-contracts.md#capability-宣告與實作) 與
[writeback ownership](../v2/gui-contracts.md#修改原則)。

## 5. 寫 guide.md 與 package 說明

把 `guide.md` 改成新實驗的 operator 文件。
寫出它掃 gain／frequency、依賴哪些 qubit／resonator 校準、何時會有 Invalid 欄位，
以及如何從 2D map 判讀。明說本篇沒有 fit 或 writeback。
不要把 Python 類別解說放進 guide，見 [operator guide 約束](../v2/measure-authoring.md#operator-workflow-guide-約束)。

原範本的 `__init__.py` 只有 package docstring。改成 power map 的說明即可，
不需要補 eager imports。若你另加公開 exports，確認它們指向新核心，且 import 時沒有硬體或背景工作副作用。
這項限制與重載範圍見 [adapter 載入契約](../v2/measure-authoring.md#目錄佈局)。

## 6. 在 definitions.py 註冊，再從 New Tab 開啟

在 `zcu_lab/definitions.py` 的 imports 加入：

```python
from zcu_lab.v2.twotone.power_map.gui import PowerMapAdapter
```

再在 `ADAPTERS` 字典加一行，保留原本的 key：

```python
"twotone/power_map": PowerMapAdapter,
```

Catalog 不會掃描 class 自動註冊。完整規則見 [adapter 載入契約](../v2/measure-authoring.md#目錄佈局)。
啟動 GUI 並開啟你自己的 entry／point，讓 alias 指向可供這個表單讀取的元件。
如果 GUI 已在執行，關閉所有 tabs 並等待操作結束，再用 Reload experiments 載入新 catalog。
修改 startup-only component 或保留的共用基礎層時，要重啟 GUI。

在 GUI 完成以下檢查：

1. 打開 New Tab，展開 `twotone`，確認有 `power_map`，原本的 `power_dep` 也仍在。
2. 選 `power_map`，確認出現 modules、gain／frequency sweep 與 averaging 輸入。
   展開 Sweep 下的 Gain，起始範圍應為 0.001 到 0.2，共 31 點。
3. 切到 Guide tab，確認顯示新 `guide.md`，沒有指回舊實驗。
4. 缺少校準時，補 entry／library 資料後按 Reset 重新捕捉來源。
   能開表單不表示 cfg 有效，也不表示已取得硬體執行授權。

若 New Tab 沒有新項目，先核對 definitions 的 import／key 與載入錯誤。
若表單無法開啟，核對新 core import、Template binding 與 alias。
下一篇會為這個 adapter 加上 [MCP recipe](04-recipe.md)。
