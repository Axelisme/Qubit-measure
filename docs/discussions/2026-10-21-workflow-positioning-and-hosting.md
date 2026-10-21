---
date: 2026-10-21
status: active
superseded_by: null
related:
  - docs/adr/0075-agent-interface-layers.md
  - docs/adr/0062-experiment-workflow.md
  - docs/adr/0063-persistence-ownership.md
  - docs/adr/0072-module-boundaries.md
  - docs/adr/0073-measure-form-derivation.md
  - docs/discussions/2026-10-21-three-role-protocol.md
  - docs/discussions/2026-10-21-recipe-interface-redesign.md
  - lib/zcu_tools/gui/app/workflows/README.md
  - zcu_lab/workflows/README.md
basis:
  - "repo@2a8a67b71：workflows 為獨立 app（scripts/run_workflows_gui.py），coherence_fluxdep 約 2,000 行，workflows/common/acquire.py 五個 run_* 自建 pulse sequence"
  - "workflow 經 adapter Form 組 cfg 的 feasibility prototype 為研究工作檔，未入庫；結果：T1 的 37 個 cfg 葉全等、兩路 acquisition IQ digest 相同、fit 差 0"
---

# Workflow 定位、承載與執行基底

## 結論

Recipe 是操作 GUI tab 的腳本，可開多個 tab、串多個實驗；workflow 是跑在 Engine 的腳本，保留一 run 一 operation、Pause／Resume、tunables revision、records 與無人值守。兩者都是腳本，輸入都以 pydantic `FormModel` 宣告，底層同一份 Wait／Ask／Signal 協定。使用者：「該統一的是使用者與作者的認知與慣例，當初是因為workflow的ui需要及時顯示累積訊息，用戶與agent也可以中途插手，所以自成語法」。

承載：workflow 由 measure-gui 承載（同一程序、同一 session、同一 RUN lease、同一 MCP server），`WorkflowWindow` 以既有可嵌入設計嵌進 measure；呈現（tab 樣式或同級容器）由 UI prototype 決定；`run_workflows_gui.py` 留作無頭與備用入口。使用者：「我也傾向B+D」。

執行基底：workflow 維持 helper 實作，不走 adapter Form、不走 core.run、不走 adapter analysis。只有 workflow 使用的實驗（freq、len_rabi、ro landscape、t1、t2echo）在其目錄內加 `workflow_helper.py` 與 `core.py` 並列，放該實驗給 workflow 用的 acquisition helper，pulse sequence 允許與 core 重複定義，維護以實驗目錄為單位。共用的是零件：cfg model、`signal2real`、fit kernel、result 型別；分析政策與接受門檻各 owner 自持。使用者：「根據我的經驗，workflow與會使用到的具體的實驗耦合度極高，特別是畫圖方式，不太適合使用core的通用流程，而且大概也不走analysis」；「sequence重複定義兩次沒有什麼不好的。core一次，旁邊的檔案，可能叫workflow_helper.py一次……要改一個實驗通常以一個目錄為單位不難」；「不要每個實驗，workflow需要的實驗就好」。

## 問題

五種腳本式定義（adapter、startup、recipe、workflow step、Notebook）形狀相近；同一個實驗的 cfg 組裝有三份（adapter Form、workflow step、Notebook）；獨立 workflows app 與 measure-gui 是兩個殼、兩套 MCP、兩套 live 圖；workflow 的 `acquire.py` 重寫與 core 相同的 pulse sequence，沒有東西保證 GUI 與 workflow 播的脈衝一致。使用者問：「workflow到底要不要經由measure-gui實現，獨立app有什麼好處」。

## 考慮過的選項與比較

### 承載

| | 基底 | 解掉什麼 | 留下什麼 |
| --- | --- | --- | --- |
| A 維持獨立 app | Engine，獨立程序 | 無 | 兩個殼、兩套 MCP、cfg 重複 |
| B 嵌入 measure-gui | Engine，同程序同 session | 兩個殼、MCP 合一 | cfg 重複、兩套圖 |
| C 步驟改走 measure tab | GUI tab | cfg 重複、兩套圖、MCP | Engine 殼只剩 Pause／Resume；無人值守綁在 tab 與 session |
| D 步驟經 adapter Form 組 cfg | Engine | cfg 重複 | 兩個殼 |

獨立 app 的好處中站得住的是 Engine 的 run 語意與無頭可跑；程序隔離只在同時開兩個 GUI 操作同一台硬體時有意義，UI 嵌進 measure 一樣能保留。採 B。

### 執行基底

| | cfg 怎麼來 | program 怎麼跑 | 代價 |
| --- | --- | --- | --- |
| helper（現狀） | 直接組 core cfg | `acquire.py` 自建 sequence | 序列兩份 |
| 經 Form（原 D） | adapter Form 無頭求值 | core.run | 要開 gui 接縫；Form 的 capture 預設 workflow 不需要 |
| 經 core.run | 直接組 core cfg | `env.run(core.run, cfg)` | 通用流程與 workflow 的畫圖、分析政策耦合 |
| 共用 `sequence.py` | 直接組 core cfg | helper 用 core 公開的序列函式 | 多一個共用函式；序列不同的步驟仍自寫 |
| **helper 搬到實驗目錄** | 直接組 core cfg | helper 自建序列，與 core 並列 | 序列重複，但維護單位是實驗目錄 |

Prototype 證明經 Form 的 cfg 與手組的 37 個葉全等、IQ digest 相同，等於白繞一圈；它也揭露 `workflow-science-no-frontends` 禁止科學層 import gui，經 Form 要開接縫（typed capability 約 80–160 行，或中立 owner 200–400 行）。使用者對經 core.run 保留意見；最後採 helper 搬到實驗目錄。

### 分析

adapter analysis 成為唯一 owner 並為 workflow 加參數，對各 owner 自持政策、共用 fit kernel。採後者。現況已是如此：workflow 的 Ramsey normalization／polarity、Echo 保留首點、LenRabi `decay=False, fit_phase=True` 都是政策層，kernel 與 `signal2real` 本來就共用。使用者：「adapter會被迫支援只有workflow需要的參數，workflow需要的t1與adapter的t1可能會隨著需求出現變化，硬是整合會難以維護」。

## 被否決的選項與理由

- C：無人值守綁在 tab 機制上，且要重寫 Engine 執行模型。
- D 經 Form：Form 的 capture 預設 workflow 不要，且要開 gui 接縫。
- 經 core.run：畫圖與分析政策耦合不合。
- 共用 `sequence.py`：同目錄重複可接受，不另設共用函式。
- 把 ctx／session／env 合成一個 capability 物件讓三種腳本共用：把 L1 tab 操作與 L3 runtime 直跑混在同一物件，推翻分層。

## 參考形狀

```python
@workflow("coherence_fluxdep", plan=CoherencePlan, tunables=CoherenceTunables, init=init_coherence)
def coherence_fluxdep(env, plan, tun, state) -> Step[CoherenceState, CoherencePoint]:
    if state.index >= len(plan.fluxes):
        return Done()
    flux = plan.fluxes[state.index]
    yield from env.set_device(plan.flux_dev, flux, unit=plan.flux_unit)   # 糖 = Wait(device op)
    freq = yield from calibrate_frequency(env, captured, state.cal)
    rabi = yield from calibrate_rabi(env, captured, state.cal)             # 假設 frequency 已接受
    if rabi.status == "ambiguous":
        answer = yield from env.ask(RabiChoice(candidates=rabi.candidates))  # Ask：run 進 paused
        rabi = rabi.pick(answer.index)
    cfg = build_t1_cfg(plan, tun.t1, state.cal, guess_us=state.t1_history.guess())
    outcome = yield from env.run(run_t1, cfg, live=plot_t1.update)         # 糖 = Wait(acquisition op)
    save_t1(outcome, env.iteration_dir / "t1.h5", snapshot=plan.snapshot)
    fit = analyze_t1(outcome.result)                                         # 自己的政策，共用 fit kernel
    state.index += 1
    return Next(record=CoherencePoint(flux, t1=fit if accepted(fit, tun.t1) else None))
```

`run_t1` 住在 `zcu_lab/v2/twotone/time_domain/t1/workflow_helper.py`。實作備註：state 以索引記進度，plan 不可變，不以 `remaining.pop()` 消耗清單；helper 帶 workflow 的階段契約（如 `calibrate_rabi` 假設 frequency 已接受），core 與 adapter 不該有這種前置假設。

## 留下的問題

- 呈現：workflow 在 measure-gui 裡是看起來像 tab 的同級容器還是 dock；measure 的 `State.tabs` 假設每個 tab 對應一個 adapter 的 Run／Analysis／Save pane，workflow 不進 `State.tabs`。
- 單一 RUN lease 與 measure 的 `global_run_active` 旗標如何對接；MCP 合併後 `status` 如何同時報 tab operations 與 workflow runs。
- `Ask` 使 run paused 時的 GUI 呈現與回答控件。
- 五個 `run_*` 的 provenance／evidence 樣板是否收成 workflow 內部 helper。

## 依據

- Prototype 的 cfg／acquisition／fit parity 與遷移估計來自固定 commit 的研究工作檔，未入庫。
- 獨立 app 的現狀與可嵌入設計見 related 的 workflows app README；Engine 的 run 語意見 ADR-0062；保存權威見 ADR-0063。
- ADR-0075 的定位段與 ADR-0062 的 effect 段依本文更新，待實作 task 落檔。
