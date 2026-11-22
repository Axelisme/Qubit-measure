# Measure adapter authoring

**讀者：** lab-author、framework-dev、agent。撰寫 Measure adapter、Form、recipe、operator guide 與校準 writeback。

**Last updated:** 2026-11-09

使用者入口見 [tutorial](../tutorial/README.md)。

## 目錄

- [設計原則](#設計原則)
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

`zcu_lab/v2/<實驗>/gui.py` 以具名參數宣告框架的 `Adapter(spec, make_form, run, guide, analysis=None, post_analysis=None)` 值，接上同資料夾的 core，供 `zcu_tools.gui.app.measure` 驅動。框架不 import 使用者套件，也不含實驗領域知識。

---

## 設計原則

實際量測的變數太多，不以通用流程或通用工具預先包辦，最多定慣例。分析步驟由作者決定，框架與 notebook 提供可組裝的小工具。遵守 KISS，一個工具做一件事，不替作者預先選分析順序或修正方式。客製化與避免強制流程的審查規則見[程式碼品質](../../docs/code-quality.md#zcu_lab-審查裁決)。

2026-10-27，使用者裁決：「由於實際測量可能遇到的變數太多了，幾乎很難設計通用的流程與工具，最多只能寫一個慣例，但實際怎麼分析應該完全能照作者的想法解析，請你抱著這個想法設計工具。比如說每個工具都要小，遵守KISS原則，這樣才好組裝。」

## 目錄佈局

實驗的 leaf layout 見[套件 README](README.md)。`../definitions.py` 明列 adapter catalog，
`../templates.py` 提供注入 Registry 的持久範本目錄。公開支援套件 `support/measure/` 提供跨 adapter
共用的 analysis、writeback 與 context mechanics，也可集中已定案且供多個 Form 共用的具名初值政策。

每個實驗在 `core.py` 定義 cfg 與 FieldMeta，在 `form.py` 定義 Qt-free Form。框架透過 adapter 的 `make_form(ctx)` 呼叫它。
core.py 的 Experiment 擁有 run 與 analyze 的計算；gui.py 只接線、解析來源、決定 writeback 與 operator guide。
分析計算包含擬合、逐 tone 迴圈、失敗隔離與結果型別。GUI 為 writeback 做的有限值檢查與摘要整理不算分析計算。
mux/onetone 的逐 tone 數值計算、outcomes 與單 tone 失敗隔離由 core 擁有。
gui.py 保留來源與 route prior 解析，以及 writeback。
`gui.py` 宣告具名參數的 `Adapter` 值。Form 從 adapter 輸入型別推導形狀；tree 只宣告初值、Slot／Template、鎖定與隱藏。
完整作者契約見 [ADR-0073](../../docs/adr/0073-measure-form-derivation.md)。

## 實驗重載

新增 adapter 必須明列於 `../definitions.py`，不靠 class 掃描註冊。組合根重載
`zcu_lab.v2` 與 `zcu_lab.definitions`，包含 package exports 與引用 core 型別的 Notebook flux picker。
`zcu_lab.templates`、`support` 下的 measure／singleshot 支援家族，以及框架 identity 保留。
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

每個實驗的 `gui.py` 是其 GUI policy 的修改入口。至少被兩個 adapter 共用的 mechanics 或已定案初值政策才放進公開支援套件 `support/measure/`。單一實驗的初值政策留在 `form.py`。

Adapter 以 `guide=Path(__file__).with_name("guide.md")` 宣告 operator guide。
框架每次 `guide()` 讀 UTF-8 檔案，回 `AdapterGuide(markdown, resource_dir)`。
`run(inp, *, context)` 接收 Form 驗證後的輸入，轉成實驗 cfg，再回傳 `RunRecord(cfg=cfg, result=...)`。
不需轉換時也在本地宣告 run 並配對同次 cfg 與 Result。額外驗證放在 run 開頭，硬體 I/O 前失敗。
Run 不重新讀目前的 point、library 或 alias。所有 Run 都需要 SoC，框架不提供額外 preflight hook。

Open／Reset 呼叫 `make_form(ctx)`，依固定來源建立 fresh input。Run 頁的按鈕叫 Defaults，RPC 仍叫 `tab.reset_cfg`。Refresh 也重建 Form 定義，
但保留 authored input、explicit selections 與既有初值，不重新套 defaults。
`ctx.has(ref)` 用固定來源選 Slot 或 Template，缺參照回 False，來源故障 raise。
單音 readout module 與 `module.readout` slot 由 resonator 擁有。單 qubit 表單透過 `resonator: $qubit.resonator` alias 讀取 `$resonator.module.readout`，沒有 slot 時使用 `Template("readout")`，不讀 qubit 上的舊 slot。
既有 point 不自動遷移，需重跑 onetone/freq 並 Apply module／slot，或手動設定 `R1.module.readout` 指向有效的 library module，否則表單使用 readout 範本。
字串是 literal，Expr 才是運算式。
初值 expression 在建立 input 時先求值並檢查欄位接受條件。具 Eval 能力的 leaf 保留原文，初開顯示 Expression；Refresh 保留原文並以新 capture 求值。
宣告 choices 的文字或數值 leaf 沿用既有初值求值能力，以可接受的 direct 值保存，不開放 authored Expr input。Literal、Enum、bool 與 InputPolicy 禁用 expression 的 leaf 不新增初值求值能力。使用者輸入的 expression 仍保留原文交一般驗證。
必要來源缺少時留 Invalid，不由 framework 補數字。實驗可在 `make_form` 明確選擇已定案的首次初值，見下節。Reset 重新捕捉並可換形狀，Run 使用畫面上的值。

Form lowering 將表單結果轉成 adapter 的具名輸入，失敗時回報欄位或模型錯誤。
Qt 呈現框架的 [Tab admission 決定](../../lib/zcu_tools/gui/app/measure/README.md#tab-admission)，另處理編輯器中尚未提交的解析錯誤。

組合根以 `Registry(templates=TEMPLATES)` 注入預設持久範本，Library 新建選單也讀這份 catalog。chain_length 與 rpi_length 的 Form factory 由 `register_all` 綁定 `registry.templates`，讓長度差初值與 Form 展開使用同一份範本。
範本儲存 Spec defaults 與 seeds，不維護另一份角色 shape／blank／ref 表。
Library、writeback 與 Inspect 也使用 gui.form／FormEditor，透過 gui.form_adapter
投影持久 Spec。Adapter 的 ModuleWriteback 直接提供 ModuleSpec proposal，不組編輯樹。
Child key 保持 key，expression 在固定 env 求成數字後才存 Library。具體 family 固定，
屏蔽欄位由生成器預設補回。提交與 draft 邊界見 ADR-0065、ADR-0008。

Adapter 如需不同於 core 的欄位，Form 直接使用具名輸入型別，轉換由 adapter.run 擁有。
Form 沒有 extras／lower 容器。T2Ramsey／T2Echo 的 detune_ratio 與 CPMG 的
min_length_us、max_length_us、length_points 都是輸入的頂層欄位。
`onetone/freq` 的 sampling 是頂層選擇，由 run 轉回 core 的 sampling_mode 與 homophasal。
Linear 輸出 homophasal=None。Homophasal 的必要參數由固定 capture 提供 fresh 初值，可覆寫；
缺值或無效值阻止提交，不從 live point 補值。Record 只保存實際實驗 cfg，不保存另一份 Form input。
`twotone/time_domain/t1` 的 uniform 隨 RunRecord 保存。False 保原 start／stop／expts 視窗，
由 runtime 在硬體量化前配置 delay；格點 collision 提示擴大 span 或減少 points，不靜默減點。

Core 用具名裝置欄位，例如 flux_dev、jpa_flux_dev。Form 在 Open／Reset 從
$qubit.flux_dev、$mux.flux_dev 或 JPA 的 flux_dev／rf_source 取得可覆寫的 direct 名稱。
缺來源留空，必填欄位報錯。Core 只以 context.devices 查 driver。Flux_unit 獨立保存，recipe 不改單位政策。

## 表單設計與預演

Form 預設依 cfg 宣告順序顯示欄位，不寫 `field_order`。
Cfg 依序放播放順序的 modules、外到內的掃描量、取樣方式、取數欄位，
見 [cfg 撰寫慣例](experiment-authoring.md#cfg-名稱順序與取數欄位)。
Adapter 使用獨立輸入型別時，預設順序由該型別決定。
只有需要調整頂層欄位時才寫 `field_order`，並在原地用一行註解說明原因。
例如 `onetone/freq` 把繼承而來的 sampling 欄位移到 frequency sweep 後。
指定欄位依 tuple 順序前移，未列欄位保留原相對順序，nested model 不變。
`tree` 的寫入順序不控制畫面。未知或重複欄位會在 Form 建構時 raise `ValueError`，
持久 Spec 的 Form 也不接受非空 `field_order`。
在 guide 依實際可見順序寫操作步驟，預演時核對繼承欄位與例外排序。

在宣告模型的 `FieldMeta` 指定可見 label／unit，照[表單文字 F1–F4](figure-style.md#表單文字)及
[單位表](figure-style.md#單位)撰寫，例如「自由演化時間 (µs)」和「驅動增益 (a.u.)」。
SweepCfg 的座標與 step 繼承 unit，點數不繼承掃描量的物理單位。
既有 cfg key 不因顯示文字改名，例如 `delays`、`delay` 與 `length` 都可顯示 Delay。
共用但物理量不同的 range 必須說明單位由哪張實驗決定。
Form 的 flux 掃描輸入使用 `flux_unit` 指定的裝置輸入單位，不用它推定 Result 或保存軸的單位。
例如 [JPA bias survey](jpa/flux_onetone/guide.md#read-the-plots) 的圖上 bias 軸與保存的 `fluxes` 軸都使用裝置原生 a.u.，不隨 Form 的 A／V／native 選項改標籤或換算。
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
Hidden 只用於 Run 會覆寫或計算的值，不用來藏起 required input。

必填欄位的 label、unit 與 description 宣告在靜態 `FieldMeta`，寫法見 [F3](figure-style.md#f3description-用一句填寫說明)。
有作者說明的必填空白錯誤會附上該欄位與父欄位的 label／description，保留原 path。
Recipe preflight 沿用這些 message 與 typed paths，不從 generic error 猜物理設定。
Readout slot found 與 form valid 必須分開判讀；slot 存在不代表 drive、GE centers 或掃描範圍齊全。
Pulse／Mux Readout 的 generator、ADC 與 classifier 契約見 [Program 的 Readout configuration](../../lib/zcu_tools/program/v2/modules/README.md#readout-configuration)。
Mux participants 只決定執行順序，未參與 tone 仍須通過驗證。

新實驗交付前，以 `QT_QPA_PLATFORM=offscreen` 預演下列情境：

1. Kickstart entry 只有接線與 nominal values，沒有 module slot。
   記錄哪些欄位有 seed、哪些欄位必填空白，以及 uncalibrated 候選。
2. 加入必要 frequency，仍不加入 readout slot。
   比較 Open、Refresh、Defaults。Refresh 保留舊輸入，Defaults 依新 capture 重新取初值。
3. 使用包含 readout slot 的 project 副本。
   分開列出 slot 是否存在，以及表單尚缺哪些 pulse、drive、timing 或 GE calibration。
4. 核對對應的 agent recipe。
   確認必填參數、單位、預設與 null 語意；以缺參情境核對 preflight 能否指出下一步。
   沒有 recipe 的表單在 guide 說明使用 GUI／RPC，不假設 recipe 覆蓋所有實驗。

每個 GUI 情境保存 get_cfg 的 input、values、errors、diagnostics 與實際欄位順序。
用 ComponentDriver 開啟 tab、操作 Refresh 與 Defaults，保存截圖並審閱可見 label、單位與錯誤。
在 guide 寫出各情境的起手步驟與仍需取得的校準。
預演不呼叫 Run、不連硬體，也不操作使用者正在使用的桌面 GUI，除非另有明確授權。
Guide 必須提醒操作員：來源修改後，按 Defaults 重取初值，確認後會丟棄表單編輯。Refresh 只更新 source／choice，保留輸入。

## Recipe 撰寫

「我會希望zcu_lab中的內容有最大化的可讀性，上手時間最短，實作內容一看就懂」

在 zcu_lab 內，可讀性優先於 lib 的強型別、防禦分支與抽象規則。衝突時選一眼就懂的寫法，
型別與驗證由 lib 在邊界承擔，不在 recipe 另寫一套驗證。
審查標準是「新人多久能看懂這個 recipe／core」，不是 class 數或型別完整性。

「加入自己的客製化修改也很自然」

改脈衝、多欄位或換分析步驟，應像改腳本，不必先理解基底或 helper 契約。
lib 基底只給可選的預設路徑，不限制 recipe 直接寫自己的步驟，也不要求套進固定流程。
新增自己的設定時，在輸入宣告欄位，再在步驟發生的位置使用它。

審查時逐欄確認 cfg／表單／recipe 欄位在實驗中的用途。沒有用途的欄位不保留。

Recipe 放在 [`../recipes/`](../recipes/README.md)。撰寫時遵守以下六項，審查裁決見
[程式碼品質](../../docs/code-quality.md#zcu_lab-審查裁決)：

1. 主體直接寫開 tab、改欄位、Run、Analyze、問答、寫回及 Save 的順序。
   選 module、設值、`tab.apply_overrides` 與 check_ready 都明寫，不藏進 helper。
2. 分析結果引用 owner 匯出的型別，不另寫 FitQuality、QualityIssue 或 summary 鏡像 model。
3. Recipe class 只宣告實驗輸入與報告，不為步驟再建 runner 或 pipeline class。
4. Helper 只做數字呈現等機械格式化。選脈衝、掃描範圍、重測與寫回判斷留在主體。
5. Review core 時逐項說明 class 與抽象的用途，讓讀者從 `run` 看見脈衝序列。
   Class 數只提示閱讀成本，不是新增或刪除 class 的配額。
6. Docstring 用使用者術語交代輸入、動作、結果與失敗。寫明單位、分析參數的效果，以及 `n → expts` 或 `points → expts` 的對應。可省略的欄位要說明省略時保留什麼、明填 null 如何原樣交給 owner。
   Input schema 與 report 的可見文字另依 [Recipe 說明 S1–S2](figure-style.md#recipe-說明)撰寫，不能代替 docstring 的契約。

以下骨架使用 `recipe_api` 的真實介面與 twotone owner 的具名 summary。
輸入繼承 `RecipeInputs`，報告繼承 `RecipeReport`。`RecipeInputs` 不宣告欄位。
Recipe 可按需宣告 `target: SingleTarget`、`reuse_tab_id: ReuseTabId` 與 `overrides: Overrides`。
`target` 指定本次開啟或重設表單所用的綁定，`reuse_tab_id` 選擇要重用的 tab，
`overrides` 在實驗欄位設好後套用額外的表單操作。
多角色可以改用自己的 target model。基底也不提供通用 library／inline 輸入。
可省略的實驗欄位用 `T | None | Keep = KEEP`，省略保留，null 原樣交 owner。
以下例子的掃描端點與 pulse length 都必填，分析選項可省略。
`Expts (count)` 和 `Length (µs)` 是顯示詞彙遷移後的目標，不是現行 UI 文字，也不改 cfg 或存檔格式。

骨架先對 drive 與 waveform command to_inline。已 inline 時 command 不改值。
這樣改長度會保留既有 waveform family，不固定換成 const，單 kind module 也照此處理。
每欄 set 完才用 `tab.apply_overrides` 套 overrides，最後 check_ready。中間可插入自己的步驟：

```python
from typing import Literal

from pydantic import Field
from zcu_tools.mcp.measure.recipe_api import (
    KEEP, Keep, Overrides, RecipeInputs, RecipeReport, RecipeSession,
    ReuseTabId, SingleTarget, recipe, report_number,
)
from zcu_lab.v2.twotone.freq.gui import TwotoneSpectrumSummary


class Inputs(RecipeInputs):
    """Sweep the RF drive frequency with a custom probe-pulse duration."""

    target: SingleTarget
    reuse_tab_id: ReuseTabId
    overrides: Overrides
    start_mhz: float = Field(description="Set Frequency › Start (MHz), the first RF frequency.")
    stop_mhz: float = Field(description="Set Frequency › Stop (MHz), the last RF frequency.")
    n: int = Field(ge=2, description="Set Frequency › Expts (count), the number of sweep points.")
    pulse_length_us: float = Field(
        gt=0, description="Set Probe Pulse › Waveform › Length (µs), the drive duration."
    )
    model_type: Literal["lor", "sinc"] | None | Keep = Field(
        KEEP,
        description=(
            "Choose the peak-fit line shape in Analyze › Model type; "
            "omission keeps it, null passes to the owner."
        ),
    )


class Report(RecipeReport):
    """Transition estimate and the inherited writeback/analysis facts."""

    freq: float | None = Field(default=None, description="Qubit frequency (MHz).")


@recipe("custom_spectrum", description="Sweep drive frequency with a custom pulse.")
async def custom_spectrum(session: RecipeSession, inputs: Inputs) -> Report:
    """Sweep, fit a transition, ask before writeback, then save.

    start_mhz and stop_mhz set the RF sweep endpoints in MHz.
    n sets the number of points via Form freq.n, which becomes cfg freq.expts.
    pulse_length_us sets the probe duration in microseconds.
    model_type chooses a Lorentzian or sinc line shape for the peak fit.
    Omission keeps the current analysis value; explicit null goes unchanged
    to Analyze for validation.
    Report the fitted qubit frequency in MHz and whether writeback was applied.
    Invalid Form values stop Run. If Analyze produces no result, report its
    failure reason and save the acquired data.
    """
    tab = session.open_tab(
        "twotone/freq", target={"qubit": inputs.target}, reuse=inputs.reuse_tab_id,
    )

    tab.command("qub_pulse", "to_inline")
    tab.command("qub_pulse.waveform", "to_inline")
    tab.set("freq.start", inputs.start_mhz)
    tab.set("freq.stop", inputs.stop_mhz)
    tab.set("freq.n", inputs.n)
    tab.set("qub_pulse.waveform.length", inputs.pulse_length_us)
    tab.apply_overrides(inputs.overrides)
    tab.check_ready()

    await tab.run()
    try:
        analysis = await tab.analyze(
            "primary", result=TwotoneSpectrumSummary,
            updates={"model_type": inputs.model_type},
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
            freq=report_number(analysis.result.freq),
            writeback_applied=applied,
        )
    finally:
        tab.save()
```

這個例子的 report 與 Analyze 摘要都呈現擬合的 qubit 頻率。若擬合值為 5800.125 MHz，兩處都寫 `Qubit frequency (MHz) = 5800.125`。
Report 從 `analysis.result.freq` 取值，core 結果與原生 JSON 摘要的 key 仍是 `freq`。
範例的 `description` 說明 report 欄位的意義，不設定 Analyze 的顯示名稱。

`FieldEdit` 是 dotted path 加上 `value`、`select`、`command` 其中一個，不帶 source；`select: null` 清空可清空的 selector。Overrides 省略為空 list，不接受 null。
`tab.apply_overrides` 依序執行、不去重；某筆失敗時報出索引與 path，前面已套用的保留，不 rollback。每次操作允許尚未 ready 的合法草稿。
`check_ready` 唯讀驗完整 Form／cfg，缺值回 needs_parameters 與所有 path／reason。
結構、來源與 readonly 錯誤仍在 mutation 時立即拒絕。
讀取與寫入都用 dotted string，數字段是完整清單的原始位置，不按 participants 重編。
若輸入代表完整 tone 範圍，先檢查與完整 tones 等長，回報缺少或多出的原始位置。
Center／span 成對提供或成對省略，不從目前 endpoints 或校準補齊。
明填 null 不做算術，原樣交 owner。全部 module scalar 省略時不轉 inline。
分析 updates 直接傳 dict，由 `tab.analyze` 送 owner 前略過 KEEP 項。
全 KEEP 等同不帶 updates。Null 原樣交分析 owner 判定，不在 recipe 偷換成省略。
參數讀取用 `tab.read_parameter`，裝置快取單位用 `session.device_unit`，不使用 RPC models。

`select`、`set`、`command`、`apply_overrides`、`check_ready`、`run`、`analyze`、問答、寫回與 `save` 沿用 RecipeTab 的公開介面。
`run` 成功後才進入 Save 的 `finally`，分析失敗仍保存 raw data 與已完成產物。
不要把整段改成無法插入步驟的 `run_all`。需要換分析時，就在 `analyze` 呼叫處換 stage、
owner 型別或分析參數，不必先修改基底。

## 執行與分析契約

GUI Load 由 `LoadService.load_result` 讀 native data、核對 `spec.tag`，再由
`Adapter.restore(stored)` 呼叫 `restore_run(stored, spec=spec)`。
Single 與 grouped spec 共用此入口，不需要建構 core。所有 adapter 都可嘗試 Load，
不支援的檔案會回報原因並保留既有結果。作者不宣告 restore hook 或 load_data 旗標。
有效 native data 的 typed cfg 不相容時保留資料與 cfg_issue；缺 cfg envelope 或壞資料仍拒絕。
平鋪 cfg 的 schema major 為 3.0；舊 2.0 cfg 不轉換，record.cfg 為 None，資料與 metadata 保留。

Load 保留 tab input 與表單固定來源，不從 loaded RunRecord.cfg 回填。分析與 writeback 另使用檔案
history 的固定來源。同 entry 有 point 時讀該 point 的現存持久值，不重建未保存的歷史參數版本。
跨 entry、EMPTY 或無 point 時來源為 None，不回退目前 aliases。Save storage 獨立固定於 Load request。
分析與 writeback 只讀 record，不讀 current Form input。

作者宣告 `Adapter(spec, make_form, run, guide, analysis=None, post_analysis=None)`。
Primary 只可為 `Analysis(fn, options=None, writeback=None, summary_lines=None)`、
`Interactive(plugin, frontend, options=None, writeback=None, summary_lines=None)` 或 None，互斥由型別表達。
第二層使用 `PostAnalysis(fn, options=None, writeback=None)`。
Capabilities 由值推導，不手填。一般 gui.py 只需 import Adapter 與 Analysis。

Primary options 的生成函式為 `options(source, env)`，回傳物件的型別提供欄位與初值。
Host 在成功 Run／Load 後建立分析表單時呼叫一次。Post options 另接已採用的 primary 結果。
MCP schema 與 JSON 重建使用同一物件型別，不重新讀歷史。編輯後的 options 交給 fn。
沒有 options 時仍能分析，表單為空。現有三個歷史預設生成器接入此機制；
現有 16 個分析時讀歷史的 wrapper 仍保留，其清理另行處理。
Figures-only 分析回 None，仍保存具名圖與空摘要，不提出 writeback。
`summary_lines(source)` 可從固定來源提供操作員文字；缺少來源值時省略該行。
GUI 和 MCP 共用這些行，不把它們寫入原生 JSON。見 [GUI contracts](gui-contracts.md#共用-lifecycle-hooks)。

Framework 依 RunRecord/spec 提供 `data_item` 與 `analysis_item`，作者不覆寫 artifact hooks。
只保存已提交輸出與歷史分析輸入，不取草稿或目前表單。缺 cfg 不補值；
準備失敗只使該 data output 失敗，不阻塞獨立的圖與分析，檔案寫入由保存流程負責。
Host 將 native 分析結果交給 `summarize_analysis`，Save／MCP 共用這個入口。
實驗端不提供 `to_summary_dict` 或原生 JSON 投影，也不繼承已退休的結果 base。
結果根須為 dataclass、pydantic model 或字串 key 的 Mapping，summary 固定為 strict JSON object。
Key 使用原生結果欄位名，recipe／workflow 依這些欄位讀取，不提供舊 key alias，也不重寫舊存檔。
Converter 將非有限數值轉為 null，Live RPC 的 `invalid` 列出路徑。
保存文件只含 pane、parameters、summary，不保存 invalid；重開存檔只看得到 null。

`RecordWriteback(..., record=...)` 只提出 adapter 組裝的 JSON object。
框架不解析內容或補座標。格式由 adapter 間的 policy 決定，見 [Record policy](#record-policy)。

完整 callback 簽名見 [adapter hooks](../../lib/zcu_tools/gui/app/measure/adapter/hooks.py)。
兩個 flux_dep adapters 共用 Qt-free flux_pick_plugin 與 Qt-only flux_pick_frontend，
service 擁有 session；adapter 不持有分析 operation 或 writeback draft。

`singleshot/ge` 的 Primary 擬合 centers、fidelity、theta、threshold 與 ge_s。
Post 使用同次 Primary 計算 radius 與 3×3 confusion matrix。
Primary 提出只有 g/e 中心的 circle，Post 提出完整 circle，各是一個完整 ModuleWriteback。
Fidelity 與 ge_s 只留在分析結果，不另寫 point 參數。
同批 Apply 固定 Primary→Post；分次 Apply 由最後一次完整 replacement 決定。
Post 後再單獨 Apply Primary 會移除 radius 與 matrix，需要 radius 的 consumer 提示做 Post。

`singleshot/amp_rabi` 與 `singleshot/len_rabi` 保存單 ADC raw IQ，支援 Pulse 或單 tone Mux。
已有 circle 與 radius 時用 saved readout 做 live G/E/Other preview。
沒有 classifier 時省略 preview，仍可 Run 與聯合 fit，不先要求另一份校準。
Backend valid 且中心、radius、matrix 全部 finite 時提出一個完整 readout ModuleWriteback。
目標由 Run capture 的 `$resonator.module.readout` 解析，缺 cfg／capture 明確失敗。
Pulse 更新 `classifier`，單 tone Mux 更新 `tones.0.classifier`。
共用中立 builder 比對其餘 materialized 欄位並保留 library 原 Spec 的 waveform key。
Adapter 不重跑 fit，不直接 Apply。

其他需要 Other 的 singleshot consumers 先從 readout cfg 要求 circle 與 radius。
作者先 `prog = builder.build()`，再以 `shot_sums(prog, [Classify("readout")])`
解析讀出 label，交給 `sched.run_acquire_summed(prog, soc, reducer=..., options=RunOptions(converter=...))`。
Timeline 使用 emit tag，未給 tag 才用 module 名稱。
Mux 軸另指定 cfg tone 位置。Circle 明列 `[g,e,other]`，缺 radius 時提示做 GE Post。
同一 converter 收到 cumulative round updates 與 final sums，不使用預設 IQ converter。
Result 保存每點 class counts，不除以 reps 或 rounds；分析與 live plot 才除以該點總計數。
需要 herald 條件平均時列出 `[Classify("herald"), IQ("signal")]`，
再以 `split_sums` 取得各 herald 類別的 counts 與平均 IQ。
零計數與未測 NaN 計數的平均 IQ 為 NaN，不發除法 warning。
Confusion correction 只讀 saved cfg；缺 cfg／classifier／matrix 就略過。
`singleshot/check` 仍允許 raw Run，分析才要求 saved circle 與 radius。
Amp Rabi 串接各 round 的 raw IQ，reset-check 累加各 round 的 class counts。
舊比例檔案缺少 counts 變數時拒絕載入，不補造計數。

[`twotone/t1_cov`](twotone/time_domain/t1_cov/guide.md) 共用單 tone `PulseReadout.Cfg`，
以 `shot_sums(prog, [IQ("pre_readout"), IQ("post_readout")])` 加總未分類的 IQ 特徵外積。
Result 保存 float64 `sums`，形狀為（delay, pre, post）。Grouped spec 從內層軸宣告，
所以先列 `post_readout_features`，再列 `pre_readout_features` 與時間。
Live 投影與 Analyze 擬合都用狀態改變機率，只有 Analyze 提出 `$qubit.t1` 寫回。

`singleshot/len_rabi`在analysis pane提供`decay: bool`，預設啟用衰減包絡；
此選擇不屬於量測cfg，不改變len Rabi的raw-IQ acquisition。
`singleshot/amp_rabi`共用Len Rabi的raw-IQ joint fit，固定無衰減／零相位，提供Initial State與相同的IQ校準writeback。

## Record policy

T1、T2Echo、T2Ramsey、TwoTone frequency 與對應的四個 mux adapter 共用下列記錄格式。
每筆 `record` 包含：

- `devices` 是 Run 開始前的完整 `RunMetadataContent(record.metadata).device_snapshot` 深拷貝，必須為 dict。
  裝置名稱與全部設定原樣保存，包含 global coil、各 qubit 的 local 線與 RF 裝置。
  DC 裝置保留 `value`／`value_unit`，RF 裝置保留 `freq_Hz`／`power_dBm`，不壓成單一值。
- `flux_frame` 複製這次量測所用 qubit 的 `flux: {offset, period: {<device>: ...}}`。Offset 是零輸出時的 Φ0，各 signed period 使用該儀器原生單位／Φ0。
  GUI adapter 在 Analyze 建立 proposal 時，從該 Run 的 `req.ctx.parameters` 取 qubit 區塊，呼叫 `record_flux_frame(flux, devices)` 取得獨立副本並核對同次 Run snapshot。
  Mux 依原始 tone index 找到 participant qubit，各自保存自己的 frame，不使用 mux group 或另一顆 qubit 的校準。
  `{offset: null, period: {}}` 表示尚未校準，不代表缺少 parameter capture。
- 分析純量沿原有名稱與單位，例如 `freq_mhz`、`t1_us`、`t2e_us`、`t2r_us` 與對應誤差。
- 頂層 `experiment` 取自同一次 RunMetadata，不再組巢狀 `run`。
  Append owner 的信封提供同一次 RunMetadata 的 `run_id`，record 不另存重複身分。
  Owner 另補 `component`、`point`、`at` 與 accepted event 的 `source`。舊 record 不轉換。

Adapter 的儀器值只讀 `req.run_result.metadata`，frame 只讀同次 Run 的固定 parameter capture。
不查目前 alias、目前 point 或 live 裝置。接受 proposal 時不重讀 frame，量測後改校準不改舊 Run 的 record。
Metadata 缺 device_snapshot key 或該值為 null 時，adapter 記 `logging.warning("no device snapshot")` 並省略 record proposal。
其他 parameter/module proposals 保留。空 snapshot dict 表示已知沒有連線裝置。
Frame 有 period 但 snapshot 缺值的儀器不貢獻，period 保留。明寫無效或 bool 值時，writer 以 `ValueError` 指名 device。
此檢查不查目前參數文件、不讀 live 值，也不限制原始量測 Save。未校準 frame 合法。
記錄 append 到 `entry.records(target)` 的 `records/components/<target>.jsonl`。
這個目錄與 `records/ledger.jsonl` 的事件信封分開，不讀舊 samples 路徑。

Coherence、overnight 與 overnight_mux 在 Start 的固定 `env.parameters` 捕捉 frame，與該次使用的 pulses 一起存入 Plan。
逐點 append 與離線重匯出只讀 Plan。Mux 的 Plan 按固定 qubit 名稱存 frame，不按 participant 的排列位置配對。
Headless caller 必須提供量測時的 frame，不能以空 mapping 代替缺失 capture。缺必要 frame 的舊 workflow manifest 會在載入 Plan 時報 validation error。
Workflow 以最後一份 acquisition 的 snapshot 核對 frame；沒有 acquisition 或 snapshot 時保留 `devices=null` 的既有政策。
Notebook 作者在建立本次 cfg 時捕捉 flux 區塊，保留該次量測使用的副本。
以 `source = exp.run(cfg)` 取得 record 後，用 `acquisition = exp.acquisition_for(source)` 取得同次 Run 開始前的儀器快照。
`source` 必須是此 adapter 最近成功 Run 返回的同一物件，不能是複製、其他 adapter、舊 Run 或 load 的 record。
Analyze 與失敗操作保留 acquisition；下一次成功 Run 取代它，成功 Load 清除它。
因此在下一次成功 Run／Load 前取出本次 evidence 與待接受的 record 欄位。
Notebook 的 Run 不把 acquisition metadata 寫入 `source.metadata`，該欄位預設為空，不從它讀快照。
`acquisition.device_snapshot is None` 時不組 record，原始量測仍可 Save；空 dict 表示已知沒有連線裝置。
快照存在時，以 `devices = deepcopy(dict(acquisition.device_snapshot))` 保留完整設定，
再用 `frame = record_flux_frame(captured_flux, devices)` 核對同次值。
`deepcopy` 從 `copy` 匯入。不得在接受時重讀目前 point、alias 或 live 裝置。
以 `saved = exp.save_run(point, source=source)` 取得同次 Run 的保存 handle，
再經 `saved.accept(records={qubit: {"flux_frame": frame, "devices": devices, ...}})` 明傳。
此處 `qubit` 是具體元件名。完整範例見 [Notebook 入口](../../lib/zcu_tools/notebook/README.md#共用實驗入口)。
框架不認得 lab 的 qubit／flux 格式，也不自動從 point 補 frame。
作者若有自己的 flux 定義，可以自行組裝 `flux_frame`，不必使用 lab helper；仍須保存量測時的定義與值。

Fluxdep 接受時透過一般 record writeback 保存原始擬合 half、integer、正的 period 與被掃 device。
Point proposal 分開提出被掃儀器的 signed period 與預設勾選的 offset，其他 periods 不變。
Offset 以擬合 half 對應 flux 0.5，扣除其他有 period 且 snapshot 有值的儀器貢獻。
被掃儀器的值由擬合 half 覆蓋一次。Note 列其他儀器值與扣除貢獻，包含值為 0 者，缺席儀器不列。
獨立 app Export params 只寫 EJ／EC／EL 與選定 period，一律不寫 component offset。
模型與版本邊界見 [ADR-0082](../../docs/adr/0082-component-flux-frame.md)。

分析 caller 可用 `resources.record_table.flux_from_frame`，逐列依 explicit flux、新 row-frame、caller fallback 換算。
Row-frame 使用 offset 與各儀器 signed periods，缺 offset 報錯並指出列與欄位，不退回 fallback。
缺 device value key 的儀器不貢獻。Nested 明寫 NaN、inf、bool 都拒絕。
Dotted 的 device value／period NaN cell 視為缺席，inf／bool 仍拒絕。
攤平無法區分缺 key 與明寫 NaN，只有 nested 保證拒絕明寫 NaN。
只有歷史 flat frame 且沒有 explicit flux 的列，作者須確認歷史校準與單位後以新 `FluxFrame` 明傳 fallback。
需要單線座標時，`project_device_coordinates` 投影成 `dev_value/dev_unit`，不代表合成了多線 flux。
兩個工具都不改 JSONL。

使用者 2026-10-25：「2mux版的比較複雜，有兩個flux儀器都可能貢獻，我的想法是，兩個儀器都紀錄，其餘都同意」。
其後裁定：「我認為框架應該對SampleWriteback保持中立無知，可能可以叫RecordWriteback也說不定，行為就是append一個格式不拘的json到紀錄的jsonl中，格式由adapter間的隱式policy決定，資訊也尤其組裝，當然文檔與註解要寫這個policy好讓人維護」。

## Operator guide 格式

Adapter guide 是 prose，不是 machine contract。內容放在實驗同目錄的 UTF-8 Markdown 檔。
Markdown 章節可自由安排，內容檢查表見 [Operator guide Gd1–Gd3](figure-style.md#operator-guide)。
Adapter 明確宣告檔案，不依 class 名稱推導路徑：

```python
from pathlib import Path
# In the Adapter declaration:
guide=Path(__file__).with_name("guide.md"),
```

同目錄的 gain／freq adapters 各自指定 `guide.gain.md`／`guide.freq.md`。
`Adapter.guide()` 回傳全文 `markdown` 與圖片解析目錄 `resource_dir`。
未宣告路徑時回「尚未撰寫」預設文字，目錄為 `None`；已宣告但缺檔時 raise。
GUI / MCP 只使用這個公開契約。

GUI 以 CommonMark 加表格顯示內容，不解析 HTML。外部連結交給系統瀏覽器。
行內 `$...$`、行內或獨立成段的 `$$...$$` 使用 matplotlib mathtext，不能解析時保留原文。
Alias 引用一律寫成 code span，例如 `$qubit.freq`、`$resonator.module.readout`。
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

Core 回傳分析品質，圖依 [R3b](figure-style.md#r3b結果旁寫-fit-品質) 呈現已回傳的品質值。
有既有判準時，由 adapter 決定 proposal 的初選與理由。Draft 複製這兩個欄位，
[writeback preview](../../lib/zcu_tools/gui/app/measure/writeback/README.md#initial-selection-and-notes) 顯示結果，不重判品質。

若只改 adapter 的寫回政策，在 scratch 用來源 `RunRecord`、分析結果與固定 context 組成 `WritebackRequest`，
呼叫公開的 `adapter.get_writeback_items`，核對 proposal 的 `selected` 與 `note`，不執行 Apply。
GUI 的 [`AnalyzeService`](../../lib/zcu_tools/gui/app/measure/services/analyze.py) 呼叫 `get_writeback_items` 後，
才把 proposals 交給 `WritebackService.create_draft`。`analysis_stage` 只呼叫 core，不經這個接縫。

[Singleshot GE adapter](singleshot/ge/gui.py) 與 [mux GE adapter](mux/singleshot_ge/gui.py) 各以
`ge_calibration_review_note` 判斷 Analyze 的初選與理由。它讀取 joint `FitQuality`，包括
`relative_parameter_errors` 裡的寬度 `s` 和中心 `sg`、`se` 的相對誤差。
`ge_post_review_note` 沿用這項判斷，並檢查 Post 混淆矩陣的 condition number。
這些初選判斷與 [GE 模型診斷](../../lib/zcu_tools/experiment/README.md#ge-模型診斷) 的
`length_ratio_max_relative_error` 分開。

ro-optimize 與 reset 的 peak-picking adapter 以 `smooth_method` 提供 `wavelet` / `gaussian`
選擇，預設 `wavelet`；`smooth` 在 GUI 標為 smoothing strength，不再只代表 Gaussian sigma。
twotone `ro_optimize/length` 的 GUI analyze param 對外命名為 `duration_t0`，表示
`SNR/sqrt(length + t0)` 的 duration-normalization term，不是 penalty strength；
`None` / `0.0` 是純 SNR peak，較小的正值比大正值更偏好短 readout。

twotone `ro_optimize` adapters 的 readout spec 一律只接受 pulse readout；writeback
產生兩層 readout writeback：`best_ro_freq` / `best_ro_gain` / `best_ro_length`
仍是 parameter context scalar；當來源 RunRecord 的 `source.cfg` 非 None 且三個 best 值可由本次
analyze result 加上固定 capture 補齊且皆為 finite number 時，adapter 同時提出
ModuleLibrary `$resonator.readout.readout_dpm`，並以 writeback `role_id="readout_dpm"` 標示這個
readout role proposal。有 capture 時，module 後新增 `$resonator.module.readout` row，值為該 capture 解析的最佳化 root。Apply module 與 slot 後，後續使用該 resonator 的表單改用最佳化讀出。無 capture 時只提出 module；缺值或 non-finite 值略過 module 與 slot writeback。
`readout_dpm` 以 `source.cfg.readout` 作為 template，將 readout
`freq` 設為 `best_ro_freq`、`gain` 設為 `best_ro_gain`、
ADC window 設為 `best_ro_length`。Waveform length 比 ADC window 長 0.03 µs，
使用 `zcu_lab.templates.READOUT_LENGTH_MARGIN_US`。Trigger offset 沿用 Run snapshot。

`onetone/freq` analyze 仍寫回 parameter context `r_f` / `rf_w` / `theta0`；當來源 RunRecord 的
`source.cfg` 非 None 且 `source.cfg.readout` 是 pulse readout 時，adapter 也提出
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
GUI 的 maximum search radius 預設為 `100.0`（MHz 軸時單位為 µs）；若已明確固定
`edelay`，不重新搜尋 branch。

---

## 校準初值政策

Open／Reset 從固定 capture 產生可覆寫的初值。具 Eval 能力的 leaf 保留有效 expression 原文，choice 則按既有能力保存 direct 值。
既有 Form 不隨來源事件重填。Refresh 以新 capture 重算保留的 expression，不改 direct input；Reset 才重建 defaults。
初值不寫回校準。Notebook 歷史常數不是相容契約。
單一實驗的政策留在 `form.py`；多個 Form 共用的 T1／Rabi 政策由
[`support/measure/`](support/measure/README.md) 的具名 helper 擁有。
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

普通 T1 的掃描終點為上述時間尺度的五倍，缺 T1 時為 150 µs。
T1 with tone 的掃描終點依序使用歷史 t1_with_tone、普通 T1、30 µs，再乘五。
只有缺值才下探下一來源；tone T1 無效時留空，不跳過它。
自然等待仍用普通 T1，不用 tone T1。這些初值不保證熱平衡或指定保真度。
Reps、rounds、shots、點數與其他特殊時序不因等待政策改動。

### Probe 與 Rabi

`qub_probe` 與 `twotone/freq` 的 probe waveform.length 公允值為 1 µs，
不依 x180／pi_len，也不新增 probe_length alias。

Amplitude Rabi 的 gain 起點維持 −0.3。終點為 `min(1.0, multiplier * gain)`，gain 取固定 capture 的 x180.gain。
Singleshot population 的 multiplier 為 4.0；twotone 與 mux IQ 為 1.2。終點以 Expr 保存，缺來源或求值失敗時留空。
有 x180 且 waveform 有持久 scalar length 時，複製捕捉的 x180，
只覆寫 waveform.length 為 1.5*pi_len*pi_gain µs。
pi_len 為 x180 waveform.length，須為非 bool 的有限正數。
pi_gain 為 x180 gain，須為非 bool 的有限數且在 (0, 1]。
Leaf 缺失／null 或無效時，長度留空，不取絕對值、不 clamp、不 fallback。
有限但無效的 gain 保留原值；gain 為 NaN 或 ±Inf 時，gain 與 length 同時留空。
沒有 x180 時，兩者使用同一個 qub_probe Template，明確設 gain=1 與 length=1 µs。

x180 為 arb 時，Amp 保留原來源 Slot、形狀與 data，不複製、不覆寫 length，
也不讀 asset。gain 起點仍為 −0.3，終點沿用上述倍率與 1.0 上限。要調長度須先換 waveform。

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
Adapter 的 run 將頂層 sampling 輸入轉回 core 的 sampling_mode 與 homophasal，widget 不特判實驗。

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

各 adapter 的 seed、裝置、單位、分析與 writeback 由 leaf guide 說明：

- [Freq](jpa/freq/guide.md)、[Flux](jpa/flux/guide.md)、[Power](jpa/power/guide.md)。
- [Auto optimize](jpa/auto_optimize/guide.md) 說明 search bounds、relative resolution 與總點數預算。
- [Flux onetone](jpa/flux_onetone/guide.md) 是 survey，不提出校準寫回。
- [Check](jpa/check/guide.md) 是 pump off/on 診斷，run 後 pump 保持 ON。

## Reset 校準實驗群（`twotone/reset/`）

各 leaf 擁有來源、純量累積與 tested-reset proposal，不在作者總覽維護第二份參數表：

| 流程 | 操作與 writeback owner |
| --- | --- |
| Single-tone | [Freq](twotone/reset/single_tone/freq/guide.md)、[Length](twotone/reset/single_tone/length/guide.md) |
| Dual-tone | [Freq](twotone/reset/dual_tone/freq/guide.md)、[Power](twotone/reset/dual_tone/power/guide.md)、[Length](twotone/reset/dual_tone/length/guide.md) |
| Bath | [Freq/gain](twotone/reset/bath/freq/guide.md)、[Length](twotone/reset/bath/length/guide.md)、[Phase](twotone/reset/bath/phase/guide.md) |
| 驗證 | [Rabi check](twotone/reset/rabi_check/guide.md)、[Singleshot reset check](singleshot/reset_check/guide.md) |

Singleshot 的初態、joint fit 與校準條件見 [experiment authoring](experiment-authoring.md#singleshot-initial-state)
及該節連到的 analysis owner／leaf guides。
