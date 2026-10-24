# Measure adapter authoring

**讀者：** lab-author、framework-dev、agent。撰寫 Measure adapter、Form、recipe、operator guide 與校準 writeback。

**Last updated:** 2026-10-23

使用者入口見 [tutorial](../tutorial/README.md)。

## 目錄

- [目錄佈局](#目錄佈局)
- [實驗重載](#實驗重載)
- [Form 與固定來源](#form-與固定來源)
- [表單設計與預演](#表單設計與預演)
- [Recipe 撰寫](#recipe-撰寫)
- [執行與分析契約](#執行與分析契約)
- [Operator guide 格式](#operator-guide-格式)
- [Readout 分析與 writeback](#readout-分析與-writeback)
- [校準初值政策](#校準初值政策)
- [Operator workflow guide 約束](#operator-workflow-guide-約束)
- [JPA 校準 family](#jpa-校準-familyjpa)
- [Reset 校準實驗群](#reset-校準實驗群twotonereset)

`zcu_lab/v2/<實驗>/gui.py` 把同資料夾的 core 包成 GUI adapter，供框架層 `zcu_tools.gui.app.measure` 驅動。Adapter 依賴框架契約 `ExpAdapterProtocol`。框架不 import 使用者套件，也不含實驗領域知識。

---

## 目錄佈局

實驗的 leaf layout 見[套件 README](README.md)。`../definitions.py` 明列 adapter catalog，
`../templates.py` 提供注入 Registry 的持久範本目錄。Private `_support/measure/` 提供跨 adapter
共用的 analysis、writeback 與 context mechanics，也可集中已定案且供多個 Form 共用的具名初值政策。

每個實驗在 `core.py` 定義 cfg 與 FieldMeta，在 `form.py` 提供 Qt-free `make_form(ctx)`。
`gui.py` 把這個 factory 接到 BaseAdapter，並擁有 Run、analysis、writeback 與 operator guide。
Form 從 core 型別推導形狀；tree 只宣告初值、Slot／Template、鎖定與隱藏。
完整作者契約見 [ADR-0073](../../docs/adr/0073-measure-form-derivation.md)。

## 實驗重載

新增 adapter 必須明列於 `../definitions.py`，不靠 class 掃描註冊。組合根重載
`zcu_lab.v2` 與 `zcu_lab.definitions`，包含 package exports 與引用 core 型別的 Notebook flux picker。
`zcu_lab.templates`、`_support` 下的 measure／singleshot 支援家族，以及框架 identity 保留。
修改這些固定依賴後要重啟 GUI。

Reload 會先確認關閉全部 tabs 並排除進行中操作，不重啟硬體，也不保留舊 tabs 執行舊版本。
Import 時不得啟動 thread、操作硬體或註冊外部 callback，任意 import side effect 不會交易回滾。
Loader 的來源檢查與 registry 建構見 [source loader 公開契約](../../lib/zcu_tools/gui/app/measure/catalog_loader.py)，
RAM snapshot、retry 與 tab 恢復由 [framework reload owner](../../lib/zcu_tools/gui/app/measure/README.md#experiment-reload) 負責。

函式內 import 仍可使用，但 Reload 不是 notebook autoreload，也不保證延後載入的版本隔離。
操作中持續編輯尚未 import 的檔案，或重載後使用 deferred/dynamic import，可能混用新舊程式。
框架不保證攔截、報錯或辨識這類靜默失敗。遇到這種情況，重啟 GUI 回到乾淨的載入狀態。
已捕捉到的錯誤仍正常回報。這項依賴完整性的限制不放寬 tab 關閉確認、RAM snapshot/retry、
操作排他或 shutdown lifecycle 的契約。

## Form 與固定來源

每個實驗的 `gui.py` 是其 GUI policy 的修改入口。至少被兩個 adapter 共用的 mechanics 或已定案初值政策才放進 private `_support/measure/`。單一實驗的初值政策留在 `form.py`。

Adapter 以 `guide_path: ClassVar[Path | None]` 宣告同目錄的 operator-facing Markdown。
BaseAdapter 每次 `guide()` 讀 UTF-8 檔案，回 `AdapterGuide(markdown, resource_dir)`。
只有需要純 preflight 的實驗才覆寫 `validate_run_request(req, cfg)`。框架把 Form 輸出原樣
交給這個 hook 與 Run，不重新驗型別或 lower。Run 不重新讀目前的 point、library 或 alias。

Open／Reset 呼叫 `make_form(ctx)`，依固定來源建立 fresh input。Refresh 也重建 Form 定義，
但保留 authored input、explicit selections 與已解析初值，不重新套 defaults。
`ctx.has(ref)` 用固定來源選 Slot 或 Template，缺參照回 False，來源故障 raise。
字串是 literal，Expr 才是運算式。
初值 expression 在建立 input 時求成 direct 值；使用者輸入的 expression 保留原文。
必要來源缺少時留 Invalid，不由 framework 補數字。實驗可在 `make_form` 明確選擇已定案的首次初值，見下節。Reset 重新捕捉並可換形狀，Run 使用畫面上的值。

持久範本目錄由組合根以 Registry(templates=TEMPLATES) 注入，也供 Library 新建選單使用。
範本儲存 Spec defaults 與 seeds，不維護另一份角色 shape／blank／ref 表。
Library、writeback 與 Inspect 也使用 gui.form／FormEditor，透過 gui.form_adapter
投影持久 Spec。Adapter 的 ModuleWriteback 直接提供 ModuleSpec proposal，不組編輯樹。
Child key 保持 key，expression 在固定 env 求成數字後才存 Library。具體 family 固定，
屏蔽欄位由生成器預設補回。提交與 draft 邊界見 ADR-0065、ADR-0008。

Adapter 如需不同於 core 的欄位，Form 明列 extras 與具名 lower，或直接回傳 adapter 自己的
型別。Runtime 與 extras 的解讀屬 adapter，框架不知道它是不是實驗 cfg。
`onetone/freq` 的 Form 以單一 tagged sampling 選擇經 extras／lower 轉回 core 的
sampling_mode 與 homophasal。Linear 輸出 homophasal=None。Homophasal 的必要參數
以固定 capture 提供 fresh 初值，可覆寫；缺值或無效值阻止提交，不從 live point 補值。
`twotone/time_domain/t1` 的 uniform 隨 RunRecord 保存。False 保原 start／stop／expts 視窗，
由 runtime 在硬體量化前配置 delay；格點 collision 提示擴大 span 或減少 points，不靜默減點。

Core 用具名裝置欄位，例如 flux_dev、jpa_flux_dev。Form 在 Open／Reset 從
$qubit.flux_source、$mux.flux_source 或 JPA 的 flux_source／rf_source 取得可覆寫的 direct 名稱。
缺來源留空，必填欄位報錯。Core 只以 context.devices 查 driver。Flux_unit 獨立保存，recipe 不改單位政策。

## 表單設計與預演

每張 Form 明列 `field_order`，不要依賴 core inheritance 的欄位順序。
可見欄位依操作流程排列：participants／tones 或 readonly roles → modules → sweep →
averaging（reps、rounds、shots）→ timing（relax_delay、initial_delay）。
可編輯 drive 與必要校準輸入放在 modules 後，掃描設定前。
`field_order` 只能排列 core cfg 的頂層欄位；nested model 維持宣告順序，extras 仍在最後。
若實驗需要例外，在 guide 說明順序與操作步驟。

同一物理量使用一致的可見 label 與單位，例如 Frequency、MHz，Delay、us，
Gain、normalized，以及掃描點數 Expts。以宣告模型的 `FieldMeta` 指定 label／unit；
SweepCfg 的座標與 step 繼承 unit，點數不帶物理單位。
既有 cfg key 不因顯示文字改名，例如 `delays`、`delay` 與 `sweep.length` 都可顯示 Delay。
共用但物理量不同的 range 必須說明單位由哪張實驗決定。
Flux 軸使用 `flux_unit`，不能把動態單位寫成固定 MHz 或 us。
Recipe docstring 明列 `n → expts` 或 `points → expts` 的對應。
Mux 的 n 是全部 tone 共用點數，不接受每個 tone 各自指定點數。

初值依來源處理。來源可推導時，用 Expr 或固定 capture，寫明來源路徑與單位。
有 operator-authored slot 時沿用 slot，並說明它不是本次已驗證的校準。
必要來源不存在時，保留必填空白，不用任意數字代填。
已授權的 literal fallback 保留原值，並說明用途、fallback 條件與操作者確認責任。
不要在機械一致性修正中改變等待時間或 fallback 政策。

對實際產生的候選初值，用 `Form.with_seed_metadata` 的逐位置
`FieldMeta(description=...)` 記錄來源路徑。只有固定常數 fallback 加上
`calibration="uncalibrated"`，description 寫明「未校準的起始值，來源：固定常數」。
從 group／resonator／qubit 參數推導的初值不加此標記，也不宣稱 measured。
它供 diagnostics 使用，不覆寫 GUI label／unit。
未改寫的 captured slot 不標成 measured；未知狀態保留未知。
用 `get_cfg.diagnostics` 的 seed_source、seed_status、uncalibrated 核對 Open／Reset 的結果。
Hidden 只用於 Run 會覆寫或 lower 會計算的值，不用來藏起 required input。

必填欄位的靜態 FieldMeta 寫可讀 label、unit 與 description。
說明用「填入 …」指出角色、單位及取得來源的前置實驗。
有作者說明的必填空白錯誤會附上該欄位與父欄位的 label／description，保留原 path。
Recipe preflight 沿用這些 message 與 typed paths，不從 generic error 猜物理設定。
Readout slot found 與 form valid 必須分開判讀；slot 存在不代表 drive、GE centers 或掃描範圍齊全。
Mux participants 只決定執行順序，未參與 tone 仍須通過驗證。

新實驗交付前，以 `QT_QPA_PLATFORM=offscreen` 預演下列情境：

1. Kickstart entry 只有接線與 nominal values，沒有 module slot。
   記錄哪些欄位有 seed、哪些欄位必填空白，以及 uncalibrated 候選。
2. 加入必要 frequency，仍不加入 readout slot。
   比較 Open、Refresh、Reset；Refresh 保留舊輸入，Reset 依新 capture 重新取初值。
3. 使用包含 readout slot 的 project 副本。
   分開列出 slot 是否存在，以及表單尚缺哪些 pulse、drive、timing 或 GE calibration。
4. 核對對應的 agent recipe。
   確認必填參數、單位、預設與 null 語意；以缺參情境核對 preflight 能否指出下一步。
   沒有 recipe 的表單在 guide 說明使用 GUI／RPC，不假設 recipe 覆蓋所有實驗。

每個 GUI 情境保存 get_cfg 的 input、values、errors、diagnostics 與實際欄位順序。
用 ComponentDriver 操作 Open／Refresh／Reset，保存截圖並審閱可見 label、單位與錯誤。
在 guide 寫出各情境的起手步驟與仍需取得的校準。
預演不呼叫 Run、不連硬體，也不操作使用者正在使用的桌面 GUI，除非另有明確授權。
Guide 必須提醒來源修改後需 Reset；Refresh 只更新 source／choice，
Reset 可能覆蓋尚未儲存的輸入。

## Recipe 撰寫

「我會希望zcu_lab中的內容有最大化的可讀性，上手時間最短，實作內容一看就懂」

在 zcu_lab 內，可讀性優先於 lib 的強型別、防禦分支與抽象規則。衝突時選一眼就懂的寫法，
型別與驗證由 lib 在邊界承擔，不在 recipe 另寫一套驗證。
審查標準是「新人多久能看懂這個 recipe／core」，不是 class 數或型別完整性。

「加入自己的客製化修改也很自然」

改脈衝、多欄位或換分析步驟，應像改腳本，不必先理解基底或 helper 契約。
lib 基底只給可選的預設路徑，不限制 recipe 直接寫自己的步驟，也不要求套進固定流程。
新增自己的設定時，在輸入宣告欄位，再在步驟發生的位置使用它。

每個 cfg／表單／recipe 欄位要能用一句使用者術語說出它在這個實驗的用途，說不出就是殘留，review 時逐欄位問。

Recipe 放在 [`../recipes/`](../recipes/README.md)。撰寫時遵守以下六項，審查裁決見
[程式碼品質](../../docs/code-quality.md#zcu_lab-審查裁決)：

1. 主體直接寫開 tab、改欄位、Run、Analyze、問答、寫回及 Save 的順序。
   選 module、設值、逐筆套 overrides 與 check_ready 都明寫，不藏進 helper。
2. 分析結果引用 owner 匯出的型別，不另寫 FitQuality、QualityIssue 或 summary 鏡像 model。
3. Recipe class 只宣告實驗輸入與報告，不為步驟再建 runner 或 pipeline class。
4. Helper 只做數字呈現等機械格式化。選脈衝、掃描範圍、重測與寫回判斷留在主體。
5. Review core 時逐項說明 class 與抽象的用途，讓讀者從 `run` 看見脈衝序列。
   Class 數只提示閱讀成本，不是新增或刪除 class 的配額。
6. Docstring 寫使用者要填的量、單位、null 語意、量測動作與分析參數的效果。
   明列 `n → expts` 或 `points → expts`，不要用框架術語取代實驗說明。

以下骨架使用 `recipe_api` 的真實介面與 twotone owner 的具名 summary。
輸入繼承 `RecipeInputs`，報告繼承 `RecipeReport`。
RecipeInputs 只提供 target、reuse_tab_id 與 overrides，不提供通用 library／inline 輸入。
可省略的實驗欄位用 `T | None | Keep = KEEP`，省略保留，null 原樣交 owner。
以下例子的掃描端點與 pulse length 都必填，分析選項可省略。

骨架先對 drive 與 waveform command to_inline。已 inline 時 command 不改值。
這樣改長度會保留既有 waveform family，不固定換成 const，單 kind module 也照此處理。
每欄 set 完才逐筆套 overrides，最後 check_ready。中間可插入自己的步驟：

```python
from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP, Keep, RecipeInputs, RecipeReport, RecipeSession, recipe, report_number,
)
from zcu_lab.v2.twotone.freq.gui import TwotoneSpectrumSummary


class Inputs(RecipeInputs):
    """Drive sweep bounds in MHz and a custom constant-pulse duration in us."""

    start_mhz: float = Field(description="Drive sweep start in MHz.")
    stop_mhz: float = Field(description="Drive sweep stop in MHz.")
    n: int = Field(ge=2, description="Sweep sample count, mapped to GUI Expts.")
    pulse_length_us: float = Field(gt=0, description="Custom drive duration in us.")
    model_type: Literal["lor", "sinc"] | None | Keep = KEEP
    plot_fit: bool | None | Keep = KEEP


class Report(RecipeReport):
    """Transition estimate and the inherited writeback/analysis facts."""

    frequency_mhz: float | None = None


@recipe("custom_spectrum", description="Sweep drive frequency with a custom pulse.")
async def custom_spectrum(session: RecipeSession, inputs: Inputs) -> Report:
    """Sweep, fit a transition, ask before writeback, then save.

    n sets Form sweep.freq.n, lowered to cfg sweep.freq.expts. Frequencies use MHz.
    pulse_length_us changes the drive duration before acquisition.
    model_type selects the existing lor/sinc fit; plot_fit controls its overlay.
    Missing Form values stop before acquisition. Omitted analysis controls retain
    owner values; explicit null is passed unchanged to the analysis owner.
    """
    tab = session.open_tab(
        "twotone/freq", target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id,
    )

    tab.command("modules.qub_pulse", "to_inline")
    tab.command("modules.qub_pulse.waveform", "to_inline")
    tab.set("sweep.freq.start", inputs.start_mhz)
    tab.set("sweep.freq.stop", inputs.stop_mhz)
    tab.set("sweep.freq.n", inputs.n)
    tab.set("modules.qub_pulse.waveform.length", inputs.pulse_length_us)
    for edit in inputs.overrides:
        tab.set(edit.path, edit.value)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary", result=TwotoneSpectrumSummary,
            updates={
                "model_type": inputs.model_type, "plot_fit": inputs.plot_fit,
            },
        )
        if analysis.result is None:
            return Report(analysis_failure=analysis.reason)
        decision = await tab.ask_writeback()
        applied = False
        if decision.accepted:
            receipt = tab.accept_writeback(decision.items)
            receipt.raise_for_error()
            applied = bool(receipt.items)
        return Report(
            frequency_mhz=report_number(analysis.result.freq),
            writeback_applied=applied,
        )
    finally:
        tab.save()
```

`FieldEdit` 只含 dotted path 與 value，不帶 source。Overrides 省略為空 list，不接受 null。
逐筆 set 不去重、不過濾 null，也不提供整批 rollback。每次操作允許尚未 ready 的合法草稿。
`check_ready` 唯讀驗完整 Form／cfg，缺值回 needs_parameters 與所有 path／reason。
結構、來源與 readonly 錯誤仍在 mutation 時立即拒絕。
讀取與寫入都用 dotted string，數字段是完整清單的原始位置，不按 participants 重編。
若輸入代表完整 tone 範圍，先檢查與完整 tones 等長，回報缺少或多出的原始位置。
Center／span 成對提供或成對省略，不從目前 endpoints 或校準補齊。
明填 null 不做算術，原樣交 owner。全部 module scalar 省略時不轉 inline。
分析 updates 直接傳 dict，由 `tab.analyze` 送 owner 前略過 KEEP 項。
全 KEEP 等同不帶 updates。Null 原樣交分析 owner 判定，不在 recipe 偷換成省略。
參數讀取用 `tab.read_parameter`，裝置快取單位用 `session.device_unit`，不使用 RPC models。

`select`、`set`、`command`、`check_ready`、`run`、`analyze`、問答、寫回與 `save` 沿用 RecipeTab 的公開介面。
`run` 成功後才進入 Save 的 `finally`，分析失敗仍保存 raw data 與已完成產物。
不要把整段改成無法插入步驟的 `run_all`。需要換分析時，就在 `analyze` 呼叫處換 stage、
owner 型別或分析參數，不必先修改基底。

## 執行與分析契約

GUI Load 由 `LoadService.load_result` 讀取 native data，核對 adapter 的 `experiment_tag`，
再呼叫 `restore(stored)`。`BaseAdapter.restore` 預設建構 `exp_cls()` 並委派
`from_stored(stored)`，回傳含 record 與 cfg_issue 的 `RestoredRun`。Single 與 grouped spec
共用此入口。需要 constructor 參數或自訂結果形式時，adapter 實作 `restore`；不支援 Load
則宣告 `load_data=False`。Hook 不做 I/O、不讀目前表單、不重捕 history，也不提供 legacy fallback。
有效 native data 的 typed cfg 不相容時可保留資料與 cfg_issue；缺 cfg envelope 或壞資料仍拒絕。
失敗在替換結果前回報，既有結果不變。

Load 保留 tab input 與表單固定來源，不從 loaded RunRecord.cfg 回填。分析與 WB 另使用檔案
history 的固定來源，同 entry 有 point 時讀取該 point 的現存持久值，並非重建未保存的歷史參數版本。
跨 entry、EMPTY 或無 point 時來源為 None，仍可純計算，不回退目前 aliases。Save storage
獨立固定於 Load request。Adapters 用 RunRecord 配對 cfg 與純 Result，分析不讀 current cfg。
OneTone／TwoTone FluxDep 的互動分析由 frontend 擁有。

GUI Save 呼叫 `data_item(key, result, *, comment)` 與
`analysis_item(key, result, parameters, *, pane)` 取得保存內容，不呼叫 adapter load/save。
Base 的 data producer 以明確 RunRecord 與 `exp_cls().persistence_spec()` 準備 DataItem；
opaque 自訂結果須覆寫 producer。Analysis producer 只取已提交輸出與當次歷史輸入，不取編輯草稿。
這些 hooks 不讀 live context、不選路徑或格式，也不寫檔。缺 cfg 不以目前表單補值，
準備失敗只使該 data output 失敗，不阻塞獨立的圖與分析。檔案寫入由 framework 保存流程負責。
確切簽名見 [adapter protocol](../../lib/zcu_tools/gui/app/measure/adapter/protocol.py) 與
[ArtifactHandover](../../lib/zcu_tools/gui/app/measure/artifacts/handover.py)；native／Labber 格式與
部分失敗政策見 [experiment owner](../../lib/zcu_tools/experiment/README.md#實驗介面與資料)。

`BaseAdapter` 在 class definition/import 時驗證 `AdapterCapabilities` 與 lifecycle method 是否
一致。`analysis=FIT` 必須實作 `analyze()` 且不得實作 interactive plugin hooks；
`analysis=INTERACTIVE` 必須實作 `make_interactive_plugin()` 與
`make_interactive_frontend()` 且不得實作 `analyze()`；`analysis=NONE` 不得實作
analyze hooks。兩個 flux_dep adapters 共用 `zcu_tools.gui.app.measure.flux_pick_plugin` 的 state/actions/commands，
以及 Qt-only `zcu_tools.gui.app.measure.flux_pick_frontend` 的 artists 與 preview；adapter 傳 FluxPickInputs 與初始數值，service 建立 session，
adapter 不持有分析 operation 或 writeback draft。`get_analyze_params()` 只在 analyze-params **無法全 default 建構**（有欄位無 default）
時才必須覆寫；params 是 None 或每欄位都有 default 時沿用 base default。
`BaseAdapter[Cfg, RunResult, AnalyzeResult, AnalyzeParams, PostResult, PostParams]` 的後四槽
預設為 None。沒有參數時省略槽位或填 None，不宣告空 dataclass。只出圖的 `analyze()`／
`post_analyze()` 回 None，summary 為空 dict、writeback 無候選，圖與空摘要照常保存。
有數值結果時才宣告結果型別；後面的槽位有值時保留前面的 None 佔位。
`post_analysis=True` 僅允許搭配 primary FIT analyze，並必須實作 `post_analyze()`。
`get_post_analyze_params()` 和 primary 一樣，只在 params 需要值時覆寫。
Post-analysis 是第二層 CPU-only 探索/比較視圖，
可選地透過 `get_post_writeback_items()` 提出獨立 writeback。`validate_run_request` 不受
capabilities 控制：Base 提供 no-op default，Protocol/Base exact signature 與 registry
conformance tests 共同鎖定 framework mandatory surface。

`singleshot/ge` 的 primary analysis 只產生 fit 產物：operator可選 backend、histogram log scale、
T1 alignment、nullable shared length ratio 與 Initial State（ground / excited，預設 ground）；advanced population priors不進GUI。所選 backend擬合
centres、`ge_s`與initial populations，右側顯示IQ distribution，並經`get_writeback_items()`提出
`fid`、`ge_s`、`g_center`、`e_center`。它的零參數post-analysis是sole confusion路徑：使用primary
fit資料計算`ge_radius`與3×3 confusion matrix，顯示完整confusion diagnostic，並經
`get_post_writeback_items()`提出`ge_radius`、`confusion_matrix`；兩組proposal分屬不同pane的
opaque draft，adapter不接觸Writeback implementation。

`singleshot/len_rabi` 的analysis維持Figure-only summary，不展開額外GUI scalar diagnostics；Figure
上方呈現measured populations與joint-fit curves，左下呈現第一點integrated-bin histogram fit，右下
呈現derived confusion matrix。完整typed numeric fit留在adapter result內。只有backend valid且
`g_center`、`e_center`、`ge_radius`、`confusion_matrix`四項全部finite時，adapter才同時提出四個
獨立`ParameterWriteback` items；任一項無效就全數略過。Adapter只投影同一次domain analysis結果，
不重跑fit、不重算stability，也不直接apply proposal。

`singleshot/amp_rabi` 與 `singleshot/reset_check` 的 `g_center`、`e_center`、`radius` 是正式 experiment cfg
欄位，Form 初值 expression 指向 `$qubit.ext.readout_cal` 的 `g_center`、`e_center`、`radius`。
缺值保持 invalid；operator 可改用 direct complex／float 值。Run 使用已解析的
cfg 校正值，由各 adapter 呼叫 core，再把同次 cfg 與純 Result 配成 RunRecord。
兩者都沿用可調整的 Reps／Rounds；Amp Rabi 串接各 round 的 raw IQ，reset-check 平均各 round 的 populations。

`singleshot/len_rabi`在analysis pane提供`decay: bool`，預設啟用衰減包絡；
此選擇不屬於量測cfg，不改變len Rabi的raw-IQ acquisition。
`singleshot/amp_rabi`共用Len Rabi的raw-IQ joint fit，固定無衰減／零相位，提供Initial State與相同的IQ校準writeback。

## Operator guide 格式

Adapter guide 是 prose，不是 machine contract。內容放在實驗同目錄的 UTF-8 Markdown 檔，
章節可自由安排。Adapter 明確宣告檔案，不依 class 名稱推導路徑：

```python
from pathlib import Path
from typing import ClassVar

# Inside the adapter class:
guide_path: ClassVar[Path | None] = Path(__file__).with_name("guide.md")
```

同目錄的 gain／freq adapters 各自指定 `guide.gain.md`／`guide.freq.md`。
`BaseAdapter.guide()` 回傳全文 `markdown` 與圖片解析目錄 `resource_dir`。
未宣告路徑時回「尚未撰寫」預設文字，目錄為 `None`；已宣告但缺檔時 raise。
GUI / MCP 只使用這個公開契約。

GUI 以 CommonMark 加表格顯示內容，不解析 HTML。外部連結交給系統瀏覽器。
行內 `$...$`、行內或獨立成段的 `$$...$$` 使用 matplotlib mathtext，不能解析時保留原文。
Alias 引用一律寫成 code span，例如 `$qubit.freq`、`$qubit.module.readout`。
其他字面 `$` 寫成 `\$`。Code span、code block 與連結網址內的 `$` 保留原文。
本機圖片以相對於 guide 目錄的路徑引用。以下是語法示意，`pulse.png` 必須由作者放在 guide 同目錄：

```markdown
![Pulse sequence](pulse.png)
```

GUI 支援 PNG／JPG，過寬圖片縮到分頁寬度，不載入網路圖片或 SVG。
MCP 只回 Markdown 原文與資源目錄的絕對路徑，不讀圖或附圖。
Agent 需要圖片時自行讀取該目錄；遠端 agent 可能無法存取 GUI 主機上的路徑。

網址使用 angle brackets 或 Markdown link 語法，例如：

```markdown
<https://example.org/>
[外部文件](https://example.org/)
```

GUI 不把裸網址當成連結，裸網址內的 `$...$` 會被當成數學式。
GUI 只把外部連結交給系統瀏覽器，相對的 `.md` 連結不解析。

Wheel 收錄 `zcu_lab` 下的 Markdown、PNG 與 JPG。新增或刪除實驗時同步維護 guide 與資源。

## Readout 分析與 writeback

ro-optimize 與 reset 的 peak-picking adapter 以 `smooth_method` 提供 `wavelet` / `gaussian`
選擇，預設 `wavelet`；`smooth` 在 GUI 標為 smoothing strength，不再只代表 Gaussian sigma。
twotone `ro_optimize/length` 的 GUI analyze param 對外命名為 `duration_t0`，表示
`SNR/sqrt(length + t0)` 的 duration-normalization term，不是 penalty strength；
`None` / `0.0` 是純 SNR peak，較小的正值比大正值更偏好短 readout。

twotone `ro_optimize` adapters 的 readout spec 一律只接受 pulse readout；writeback
產生兩層 readout writeback：`best_ro_freq` / `best_ro_gain` / `best_ro_length`
仍是 parameter context scalar；當來源 RunRecord 的 `source.cfg` 非 None 且三個 best 值可由本次
analyze result 加上 current parameter context 補齊且皆為 finite number 時，adapter 同時提出
ModuleLibrary `readout_dpm`，並以 writeback `role_id="readout_dpm"` 標示這個
readout role proposal；缺值或 non-finite 值只略過 module writeback。
`readout_dpm` 以 `source.cfg.modules.readout` 作為 template，將 readout
`freq` 設為 `best_ro_freq`、`gain` 設為 `best_ro_gain`、
waveform length 設為 `best_ro_length + READOUT_DPM_PULSE_TAIL_US`（0.1 us tail）、
ADC readout length 設為 `best_ro_length`。

`onetone/freq` analyze 仍寫回 parameter context `r_f` / `rf_w` / `theta0`；當來源 RunRecord 的
`source.cfg` 非 None 且 `source.cfg.modules.readout` 是 pulse readout 時，adapter 也提出
ModuleLibrary `readout_rf`，並以 writeback `role_id="readout"` 標示它是 Pulse
readout role 的 proposal；`readout_rf` 是 target name，不是新的 role id。
`readout_rf` 以該 snapshot readout 作為 template，只用 fitted `r_f` 覆寫
平鋪的 `freq`，gain、waveform length/style、channels、ADC
readout length、trigger timing 等欄位都沿用 snapshot。`source.cfg is None` 或
readout 不是 pulse readout 時，module writeback graceful skip，只保留 parameter items。

`onetone/freq` 的 `fit_bg_amp_slope` 預設開啟，`fit_bg_phase_curvature` 預設關閉；
兩者是獨立選項。fake one-tone 的兩個選項預設都關閉。只有啟用的項才進入
raw-complex refinement；未啟用項固定為零。若 Hanger fit 的 `Qi=None`，consumer 依
`qi_status` 呈現診斷，不將它當成數字，也不因 derived Qi 不可解釋就否決整個 fit。

`onetone/freq` 的 electrical-delay analyze policy 提供 `auto` / `calibrated` / `manual`。
`auto` 優先讀取 parameter context 中與本次 pulse generator/readout channel 相符的
`res_edelay_calibration` prior，缺少時以 bounded adaptive global search 建立 branch；`calibrated`
要求相符 prior，`manual` 要求 finite seed。prior/manual 都只固定 branch，後續仍作 local
refinement。只有實際參與 fitting（移除首尾點後）的 nonuniform grid 能自行辨識 absolute
branch；uniform grid 若沒有可信 seed，不會把 local alias 提案寫成 calibration。可持久化
的結果以單一 compound parameter writeback item
`res_edelay_calibration = {edelay, res_ch, ro_ch}` 保存，讓 delay 與 route identity 的選取
all-or-none。只有可辨識 pulse route 的 run snapshot，且有可信 seed 或 fitting grid 為 nonuniform，
才提出此 writeback。沒有 cfg snapshot 的舊資料仍可分析，不提出無 route 的 calibration。
worker 提出 proposal，`ContextService` 在 preview/apply 後負責 parameter context 寫入。
GUI 的 maximum search radius 預設為 `100.0`（MHz 軸時單位為 us）；若已明確固定
`edelay`，不重新搜尋 branch。

---

## 校準初值政策

Open／Reset 從固定 capture 產生可覆寫的 direct 初值，既有 Form 不隨來源變動重填。
初值不寫回校準。Notebook 歷史常數不是相容契約。
單一實驗的政策留在 `form.py`；多個 Form 共用的 T1／Rabi 政策由
[`_support/measure/`](_support/measure/README.md) 的具名 helper 擁有。
Form 明確選擇倍率或場景並組結構，不把校準政策放進 generic framework。

### T1 與等待

T1 有效值是非 bool 的有限正實數。Leaf 缺失／null 或沒有 capture 時，
以 30 µs 作首次時間尺度。已存但無效的值讓衍生必填欄位留空，Form 仍可開啟。
Alias 未綁、namespace／path 錯誤與來源讀取故障保留 owner 錯誤，不當成缺值。
不新增單次人工估計框架，也不把 30 µs 寫回為校準 T1。

等待初值的適用範圍如下。Active reset 不自動縮短等待。

| adapter 範圍 | relax_delay 初值 |
| --- | --- |
| Rabi 4、coherence 7、gate characterization 6、singleshot／MIST／AC Stark 7、readout optimization 5、reset calibration 9、CKP 1，共 39 個 | 5*T1，缺 T1 時為 150 µs |
| JPA freq、flux、power、auto_optimize，共 4 個 qubit SNR adapter | 1*T1，缺 T1 時為 30 µs |
| Twotone freq、power_dep、flux_dep，共 3 個頻譜 adapter | 維持 1 µs |
| Lookback、Onetone freq／power_dep／flux_dep、Mux Onetone、JPA flux_onetone／check，共 7 個 | 維持各自原值 |

普通 T1 的 sweep.stop 為上述時間尺度的五倍，缺 T1 時為 150 µs。
T1 with tone 的 sweep.stop 依序使用歷史 t1_with_tone、普通 T1、30 µs，再乘五。
只有缺值才下探下一來源；tone T1 無效時留空，不跳過它。
自然等待仍用普通 T1，不用 tone T1。這些初值不保證熱平衡或指定保真度。
Reps、rounds、shots、點數與其他特殊時序不因等待政策改動。

### Probe 與 Rabi

`qub_probe` 與 `twotone/freq` 的 probe waveform.length 公允值為 1 µs，
不依 x180／pi_len，也不新增 probe_length alias。

兩種 Amplitude Rabi 的 gain 都從 −0.3 掃到 1.0，不依 x180.gain 縮放。
有 x180 且 waveform 有持久 scalar length 時，複製捕捉的 x180，
只覆寫 waveform.length 為 1.5*pi_len*pi_gain µs。
pi_len 為 x180 waveform.length，須為非 bool 的有限正數。
pi_gain 為 x180 gain，須為非 bool 的有限數且在 (0, 1]。
Leaf 缺失／null 或無效時，長度留空，不取絕對值、不 clamp、不 fallback。
有限但無效的 gain 保留原值；gain 為 NaN 或 ±Inf 時，gain 與 length 同時留空。
沒有 x180 時，兩者使用同一個 qub_probe Template，明確設 gain=1 與 length=1 µs。

x180 為 arb 時，Amp 保留原來源 Slot、形狀與 data，不複製、不覆寫 length，
也不讀 asset。gain 掃描仍為 −0.3～1.0。要調長度須先換 waveform。

兩種 Length Rabi 都從 0.03 µs 掃到 4*pi_len。
pi_len leaf 缺失／null 時，上限為 1 µs；值存在但無效時，上限留空。
有 x180 時保留其 Slot 與 gain；沒有時用同一個 qub_probe Template，gain=1。
arb 沒有持久 scalar length，依缺 leaf 規則用 1 µs 上限，保留來源與 gain。

以上分類只針對 owner 提供的 validated capture。
Capture 前已被 schema 拒絕或轉型的資料，交來源 owner 處理。
Form 不重讀原始 YAML、不吞 ValidationError，也不自行拆 key 拼 cfg。
Nested copy 由 Form adapter 擁有，不讀 asset，不改來源或無關 child key。

### Lookback 與 Onetone 採樣

Lookback 的 Form 不提供 Reset 與 Init pulse；core 保留兩欄。
Resonator 與 mux group 各自擁有頂層 time_of_fly，單位 µs；Lookback 初值與 writeback 使用 `$resonator.time_of_fly`。
Onetone timing 讀 resonator 的頂層值，mux timing 讀 group 的 `$mux.time_of_fly`，不從 ext 或另一條路徑補值。
缺 time_of_fly 時，trig_offset 初值為 0 µs，ro_length 維持可編輯的 1.5 µs。
有效 TOF 是非 bool 的有限實數且 ≥ 0；重測使用 TOF−0.1 µs。
即使結果為負也原樣帶入，不 clamp、不改 core 限制。
已存但無效的 TOF 讓 trig_offset 留空，來源故障保留原錯誤。
0 µs 不寫回為 TOF 校準。訊號抵達時間加 pulse 長度超過 1.5 µs 時，操作員須加長視窗。

Onetone/freq 只有一個 Linear／Homophasal 選擇，不另設 Enabled。
Linear 不提供 Homophasal 參數，轉回 core 的 homophasal=None。
Homophasal 的 r_f、rf_w、theta0 必填，分別以 resonator.freq、resonator.kappa、
resonator.ext.theta0 作初值。缺來源或無法求值時留空。
可求值但無效的值保留並顯示驗證錯誤，阻止提交，由 core 驗證，不新增 adapter 驗證。
實驗透過 extras／lower 轉回 core 的 sampling_mode 與 homophasal，widget 不特判實驗。

## Operator workflow guide 約束

`run-measure-gui` skill 與各 adapter guide 描述 operator 流程時，把 mock / real
量測都當黑盒：不假設已知 flux 位置，不用 simulator truth、FakeDevice 內部真值或
predictor 輸出取代掃描與看圖判讀。

標準 bring-up 順序是：`lookback` → `onetone/freq` → `onetone/flux_dep` 找
`flx_int` / period（視 2D map 判讀）→ 將 flux device 移到 `flx_int` →
在該 flux 重跑 `onetone/freq` → `twotone/freq` 寬掃 → `twotone/freq` 窄掃 →
agent / user 審核 figure 與 writeback preview → 後續 Rabi / T1 / T2。

`onetone/freq` 第一次通常用 `linear` sampling 建立 `r_f` / `rf_w` / `theta0`，並在有
pulse-readout snapshot 時提出 `readout_rf`；這些 writeback 齊全後，可切到 `homophasal`
sampling，讓同一個 start/stop/expts 掃描在 resonator circle phase 上等距。

`twotone/flux_dep` 是 readout 與 `q_f` 可信後的後續 qubit model mapping，不是早期找
flux 的工具；readout/qubit 參數還沒處理好時通常看不到可用 arc。Writeback 責任留給
agent / human 判讀，guide 不暗示用自動 fidelity gate 代替判斷。

---

## JPA 校準 family（`jpa/`）

JPA（Josephson parametric amplifier）校準由六個 adapter 構成單一可發現的
family：startup catalog（`../definitions.py` 的 `ADAPTERS`）依 bring-up 順序註冊，generic
tab creation 與 remote adapter listing（`view.adapter_list`）因此直接列出全部六個名稱：

| 順序 | adapter | 角色 |
| --- | --- | --- |
| 1 | `jpa/freq` | 找 JPA pump frequency（FIT analysis，寫回 `$jpa.pump_freq`） |
| 2 | `jpa/flux` | 找 JPA flux sweet spot（FIT analysis，寫回 `$jpa.flux`） |
| 3 | `jpa/power` | 找 JPA pump power（FIT analysis，寫回 `$jpa.pump_power`） |
| 4 | `jpa/auto_optimize` | 聯合最佳化 flux/freq/power（FIT analysis，一次提出`$jpa.flux`、`$jpa.pump_freq` 與 `$jpa.pump_power`） |
| 5 | `jpa/flux_onetone` | flux × readout frequency 2D survey（`analysis=NONE`，無 writeback） |
| 6 | `jpa/check` | pump off/on 比較診斷（figure-only analysis，無 writeback） |

JPA 表單從 `$jpa.rf_source` 與 `$jpa.flux_source` 取得 `jpa_rf_dev` 與
`jpa_flux_dev` 的 direct 初值，使用者可覆寫。Core 以具名欄位從 context.devices
取得 driver，RF 操作使用 `set_freq`／`set_power`／`set_output`，flux 操作使用 `set_flux`。六個 adapter 都顯式暴露 `reps`、
`rounds`、`relax_delay`。`reps / rounds` 維持原值，`freq` / `flux` /
`power` 是 `10000 / 1`，`auto_optimize` 是 `1000 / 1`，`flux_onetone` 是
`100 / 10`，`check` 是 `1000 / 5`。`initial_delay` 維持 GUI 隱藏並使用 core
的 `1.0 us` default。

四個 qubit SNR adapter（`freq` / `flux` / `power` / `auto_optimize`）的
`relax_delay` 初值為 `1*T1`，不是 Notebook 固定常數。Form 透過
`_support/measure/ctx_helpers.py` 的 `t1_initial_time`，從 Open／Reset 的固定
capture 取得 `$qubit.t1`。有效 T1 是非 bool 的有限正實數。T1 leaf 缺失／null
或沒有 capture 時，使用 `30 us` 公允時間尺度，等待初值為 `30 us`。
已存但無效的 T1 讓等待欄位留空，不回落公允值。Alias、path、namespace 或來源
讀取故障維持 owner 的原錯誤，不當成缺值或以留空掩蓋。初值可覆寫，不寫回校準。
兩個純反射 adapter 不套 T1 政策，`flux_onetone` 的 `relax_delay` 維持
`0.1 us`，`check` 維持 `0.5 us`。這些是可修改的操作 seed，不是跨 setup 的
最佳 SNR 或安全保證。

`auto_optimize` 的三軸 `start` / `stop` 是 search bounds；各 sweep 的 `expts`
是會影響 phase-1 flux-grid 與 per-slice budget allocation 的 relative resolution
hints，不是 Cartesian sample count，總 measurement budget 仍由 `num_points` 表達。
JPA flux 在六個 adapter 一律以中性 device value／`a.u.` 語彙呈現（sweep label、
liveplot 與 canonical persistence 一致，無 `1e3` 縮放或 physical-unit 宣稱）。
capabilities 一致：四個校準 adapter 是 `analysis=FIT` 並提出
`ParameterWriteback`，survey/diagnostic 兩個 adapter 不提供 writeback。每個 adapter
的 operator guide 都以現在式說明 seeded bounds 是 bring-up defaults 而非 certified
safety limits，operator 必須先 review device 與 sweep 再 run；`jpa/check` 特別警告
run 結束時 pump output 保持 ON。真實硬體 acceptance 由 operator 在 device/sweep
review 後執行，不屬於 adapter code。

---

## Reset 校準實驗群（`twotone/reset/`）

對應 notebook `single_qubit.md` 的三種 reset 校準流程。使用者可以重複或不按順序執行。
五個參數實驗把本次分析值累積到 qubit ext。只有指定 finisher 提出模組。

| 流程 | 校準 adapter | Finisher | 固定 library root |
| --- | --- | --- | --- |
| single-tone sideband | `freq`、`length` | `freq` | `$qubit.reset.reset_10` |
| dual-tone | `freq`、`power`、`length` | `power` | `$qubit.reset.reset_120` |
| bath cavity-assisted | `freq_gain`、`length`、`phase` | `phase` | `$qubit.reset.reset_bath` / `reset_bath_e` |
| 共用驗證 | `check`，RabiCheck，contrast / residual analysis | 無 | 無 |

`reset/check`的analysis以reset前分支決定共同IQ投影與Rabi基頻，擬合三條gain sweep
分支；最後一條包含二次諧波。摘要提供前後半峰對峰振幅、相對contrast、相位差、
reset-only殘餘振盪與offset、二次諧波振幅、各分支residual RMS，圖中保留原始資料、
擬合線與殘差。無法解析的相位為`None`；頻率無法解析或資料不足時明確失敗。
這些平均IQ指標不等於reset fidelity，也不唯一識別coherence或population機制；不提出
writeback。cfg只設定一個`rabi_pulse`，兩次pulse使用相同波形與gain。

`singleshot/reset_check`使用硬體gain／branch sweep，保存G/E populations；
analysis可讀取confusion_matrix修正，回報reset-only平均／最大excited及最大Other，
不提供IQ校準writeback。Other不是校準後leakage，reset population不是reset-channel fidelity。

`singleshot/amp_rabi`使用硬體gain sweep保存raw IQ。Reps是每gain每round的shots，
Rounds的shots串接保存；不使用Shots或Shots per batch。本次resolved cfg的g_center、e_center、
radius凍結在snapshot中供live分類，analysis與Len Rabi共用joint fit及三面板診斷圖，重新估計
IQ centers與confusion matrix。Amp固定無衰減／零相位，提供Initial State，保留振幅、
頻率與pi/pi2 gain摘要；四項校準writeback與Len Rabi共用有效性檢查。
舊population-only檔案不符合raw-IQ axes契約，無法據此重建shots或執行joint fit。

### cfg seed 與純量累積

Open／Reset 建表單時，form.py 從固定 capture 的 `$qubit.ext.*` 求初值。
SingleToneFreq 用 reset_f ±1.5 reset_fwhm 建立 201 點頻率窗。SingleToneLength 用
reset_f。DualToneFreq 用 reset_gain1/2 和 reset_f1/2，兩個頻率軸各為中心 ±50 MHz、
201 點。DualToneLength 用兩頻率及兩 gain，DualTonePower 用兩頻率。
BathLength 與 BathPhase 用 reset_bath_freq/gain。BathFreqGain 保留 resonator.freq 與
qubit.ext.rabi_freq 的原 seed。Missing、null 或 invalid scalar 沿 Form 驗證失敗，
不代用 fallback 數字。

本次分析值直接形成 `ParameterWriteback`。五個 caller 保留以下順序：

| Caller | `$qubit.ext.*` targets |
| --- | --- |
| SingleToneFreq | `reset_f`、`reset_fwhm` |
| DualToneFreq | `reset_f1`、`reset_f2` |
| DualTonePower | `reset_gain1`、`reset_gain2` |
| BathFreqGain | `reset_bath_gain`、`reset_bath_freq` |
| BathPhase | `reset_bath_max_phase`、`reset_bath_min_phase` |

### Run-tested finisher

SingleToneFreq、DualTonePower 與 BathPhase 使用該 Run 的完整
`source.cfg.modules.tested_reset`。`module_writeback` 從 Run 的 module dump 建立隔離的
持久 Spec，adapter 只替換本次分析欄位，再提出 `ModuleWriteback(proposal=spec)`。
這份來源已是 materialized runtime cfg，child pulse／waveform 因此成為 inline Spec。
Adapter 不讀 ext、舊 module 或目前 point/alias 補欄位。手動修改後實際測過的頻率、gain、波形與其他 schema 欄位都保留。

- SingleToneFreq 只換 `pulse_cfg.freq`。FWHM 只寫 ext，不是 reset module 欄位。
- DualTonePower 只換 `pulse1_cfg.gain` 和 `pulse2_cfg.gain`，保留實測兩頻率。
- BathPhase 分別 materialize ground/excited 兩棵獨立樹，只換 `pi2_cfg.phase`。
  Ground 用 max_phase，excited 用 min_phase；兩者保留實測 cavity frequency/gain。

Module targets 使用表中的固定 roots，不經方案 slot 選目的地。已有 root 更新，缺則建立。
Finisher 在 module rows 後提出 `$qubit.module.reset` 的 `ParameterWriteback`。
值由同份 capture 的 `resolve_path` 轉為具體 root，例如 `Q1.reset.reset_10`。
Bath 預設 ground，使用者可以在 draft 中改選 excited。使用者也可以只 Apply library rows，
不選 slot row。逐列 Apply 沿既有 accepted 與成功前綴政策，不新增聯動選取或跨檔交易。

### 實驗責任與不足契約

三個 length adapter 不擬合參數，返回合法空 proposal collection。它們不提出
cfg-only module。DualToneFreq 與 BathFreqGain 只提出純量。框架不提供累積或齊全檢查。
完整合法 tested_reset 不因 ext 缺少而略過 module。公開 nullable cfg/capture 為 None
時，finisher 只返回純量，不讀目前全域來源。Source、alias、lowering、I/O 與 Apply
錯誤仍傳播，不轉成空 proposal 或原因列。

保留 downstream 的 reset/readout/x180/x90/probe slots。Reset 校準不讀
sideband/dual_tone/bath_g/bath_e 方案 slots。Migration profile 將舊 reset 純量映到上述
ext 名稱，resetf_w 映到 reset_fwhm；舊 reset modules 映到四個固定 roots，不猜
module.reset 的目前方案。Mapping revisions 為 fluxonium 1.8、transmon 1.9。

Bath `freq_gain` 保留 single-role 3D HDF5。Tomography phase 是同一 Result 的內部
sweep axis，不拆成四個 phase-resolved sidecar，也不改 Save 政策。

三個 singleshot 分析（ge / len_rabi / amp_rabi）皆提供 `Initial State`，表示 probe / swept drive pulse 之前的主要狀態；Rabi 描述 pulse 前狀態，不是第一個掃描點；啟用 phase offset 時也不等同於模型外推的零 length population。此參數只影響分析，不改量測 cfg 或 raw-IQ persistence。GE primary result 保存使用的初態，Post-Analysis 的 radius、confusion matrix 與繪圖均沿用該 snapshot，不讀取尚未重新分析的表單值。

GE 的主分析與 writeback/post 邊界拒絕非有限 centers/width、重合 centers 與不合法 populations。Optimizer 未收斂不會回傳 initial guess 當作 calibration；失敗不產生新的校準 proposal。


`twotone/rabi/len_rabi` 與 `singleshot/len_rabi` 的 analysis pane 都提供
`Fit phase offset`（`fit_phase: bool = False`）。一般 Len Rabi 啟用後自由擬合相位；
singleshot 在所選 Initial State 方向附近擬合 ±90° offset，結果圖顯示相位。
兩者都可獨立切換 decay；此參數只影響 analysis，不改 acquisition 或 raw data 格式。
一般 Len Rabi 的預設由原本自由 phase 改為固定 0°/180°，校準長度 writeback 沿用
本次模型的結果。Singleshot 保持 Figure-only summary，typed fit result 另提供
`phase` 與 `zero_length_populations`；四項 IQ calibration writeback 仍須通過既有
backend validity 與 finite calibration gate。

`t2ramsey`與`t2echo`的analysis pane提供`Fit phase offset (fringe only)`
（`fit_phase: bool = False`）。開啟時自由擬合fringe phase，關閉保留固定相位；
Ramsey停用Fit fringe或Echo選擇decay時，此選項不影響分析。既有T2與detune相關
writeback沿用本次分析結果，量測設定與資料格式不變。
