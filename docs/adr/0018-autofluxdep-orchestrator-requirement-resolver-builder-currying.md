# 0018 — autofluxdep 的 orchestrator 是純需求解析器；三介面（requires/provides/produce）+ Builder 柯里化統一 Node 與 Service

**狀態：** accepted（2026-06-05）。
**關聯：** autofluxdep-gui 的 grilling 設計（見 `lib/zcu_tools/gui/app/autofluxdep/CONTEXT.md`）。承 autofluxdep 的「Node 不是 Task、用宣告式依賴取代 cfg_maker walrus 鏈」基調。

## 背景

autofluxdep 的 sweep 要把每個量測 Node 的依賴（如 qubit_freq 要 `predict_freq`）解析好餵進去。設計初期，orchestrator 被當成「胖驅動」：它持 Tools、跑 run_body、管 figure/Plotter/Result/derivations，還在 pre_point 主動 `predictor.predict_freq(flux)` seed `predict_freq`、`predictor.update_bias(...)` 校正。

grilling 中用戶指出兩個耦合 bug：

1. **predictor 預設存在 = 假設 workflow 一定有 qubit_freq**（tools 預載 predictor）。
2. **orchestrator 出現實驗認知**：它懂 `predict_freq` 這個量、懂要從 predictor 算它、懂校正——而那是 qubit_freq 的領域知識，不該洩漏進通用層。

推導中一度想「讓實驗類直接實作 Provider Protocol、加一個 produce」統一，但這讓 produce 要嘛簽名胖（帶 soc/result/round_hook/figure），要嘛把執行環境/狀態洩漏進 orchestrator 看的介面或實驗類的 self。收斂到 **Builder + 柯里化**：實驗類是 Builder（工廠），每 flux 點 `build_node(環境)` 生成一個短命 Node，環境被閉包進 Node，`produce` 只暴露依賴 in/Patch out。

## 決策

**orchestrator 瘦成純需求解析器，只看 `requires` / `provides` / `produce` 三介面。執行單元是一個短命 Node（帶 `produce`），由每實驗一個的 Builder 每 flux 點生成、把執行環境柯里化閉包進去。Node 與 Service 都是 Builder 生成、都滿足同一 `produce`，orchestrator 不區分。**

### 1. orchestrator 只解析「需求」，不解析「順序」

執行順序 = 使用者在 workflow 列表的顯式排序，orchestrator 照跑（無拓撲排序、無 `path`）。orchestrator 唯一的事是**需求解析**：跑到一個 Provider 時，把它的 `requires` 對當前 info/module 狀態投影成 snapshot（latest-available + skip/fallback）。它**不懂** 畫圖、tools、acquire、fit、Result——這些在 orchestrator 之外的**執行層**（driver / UI）。

### 2. 三介面（requires/provides/produce）+ Builder 柯里化

orchestrator 只看三件事：`requires`、`provides`、`produce`。執行單元是一個 **Node**（帶 `produce(snapshot) -> Patch`）。Node 不是手寫的、也不是實驗類本身，而是由一個 **Builder**（每實驗一子類、無狀態）**每 flux 點生成**：

- 執行層 Run 開始呼叫 Builder 的 sweep-lived 工廠（`make_init_result` / `make_plotter`）造好 Result / Plotter（活整個 sweep）。
- 每 flux 點呼叫 `build_node(這點的 snapshot / soc / Result / round_hook / Plotter / tools)`，Builder 把這些**環境閉包進**回傳的 Node。
- Node 的 `produce` 因此只暴露「依賴 in / Patch out」——環境不洩漏進 orchestrator 看的介面。

orchestrator 對所有 provider 一律 `node.produce(snapshot)`，**零 isinstance、不區分 Node/Service**。Service（predictor）也是一種 Builder，差別只在 `build_node` 閉包的環境少（無 soc/Result/round_hook，純算），生成的 Node 的 produce 純算 `predict_freq`/`cur_m`——**不偽裝成量測 Node、不 stub 不合身介面**。

### 3. predictor 不洩漏

`predict_freq` / `cur_m` 是 predictor Service **provides** 的 key；qubit_freq 用一般 `requires` 讀。orchestrator 不懂 `predict_freq` 是什麼，只看到「某 provider provides 它、qubit_freq requires 它」。predictor 的校正（`calibrate(flux, measured_freq)`）是 Service 的方法、由 qubit_freq 量完觸發，**不是 orchestrator 做**。某 workflow 沒有 require predictor 的 key，predictor 就不載——orchestrator 不預設任何 provider 存在。

## 拒絕的替代方案

- **orchestrator 當胖驅動**（持 tools/figure、主動 seed predict_freq、做校正）：洩漏領域知識、預設 predictor 存在。否決——orchestrator 必須只見 provide/require。
- **只有 Node 是 provider，Service 走別的依賴來源**（service-query 依賴）：讓「被 provide 的 key」分兩種來源、依賴解析要分流，徒增複雜。三介面 + Builder 統一更乾淨。
- **實驗類直接實作 produce（不經 Builder）**：produce 簽名要嘛胖（帶 soc/result/round_hook/figure）、要嘛把環境/狀態洩漏進 orchestrator 介面或實驗類 self。Builder 柯里化把環境閉包進 Node，produce 保持極窄。
- **Service 偽裝成量測 Node 子類**：要 stub 掉一堆不合身介面（build_cfg/make_plotter/round_hook）——與 runner 偽裝 Task 同錯。

## 後果

- orchestrator 變薄：只做需求解析（`project_snapshot` 那類）+ 對每個 provider 呼叫 `produce`，不持 tools/figure、不跑領域 seed/校正。執行（build_node/acquire/fit/填 Result/重畫）、tools、figure 全移到執行層。這是對現有偏胖 orchestrator 的**實質重構**，非新增。
- 新增實驗只要寫一個 Builder（宣告 provides/requires + 工廠 + build_node），orchestrator 不動。
- 未來看到 Service 不是量測 Node 卻能被依賴解析、或 orchestrator 不懂 `predict_freq`，本 ADR 解釋為何如此切。
