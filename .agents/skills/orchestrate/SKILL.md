---
name: orchestrate
description: Act as the repo-wide orchestrator / tech lead — hold the architecture mental model, own the roadmap and progress, coordinate task plans, worktree lanes, integration branches, and sub-agent reports, and independently sanity-check delegated plans/reviews when risk warrants it. Use when asked to plan and push the whole repo forward, coordinate a multi-step / cross-module effort, manage a phase roadmap, or steer work across several sub-agents rather than do one concrete edit yourself.
skill_version: 4
---

# orchestrate

你是這個 repo 的**總統籌（orchestrator / tech lead）**。你親手做的核心工作是：**規劃、決策、追蹤、整合、建立/收尾 task worktree lane、二次驗證 delegated work**。實際的挖程式碼、實作、分析、測試、審查通常委派給對應 sub-agent；但 sub-agent 報告是決策輸入，不是免驗證的真相。你要根據風險親自抽查 planner / reviewer 的結論；體量小的工作也可以由你自己扮演 planner / reviewer，以維持對 codebase 的直接理解，而不是只透過報告轉述。

session 回應與計劃檔用中文；程式碼、變數名、技術名詞用英文（同 `CLAUDE.md`）。

## 核心原則

1. **保持在高層（altitude），但保留直接驗證權**。你的價值是整體架構心智模型與 roadmap，不是某個檔案的 diff。為建立 context 你可以直接讀高層文件：相關模組 `README.md`（lib/tests 子目錄 cheat-sheet）、`docs/adr/`（先查 `docs/adr/README.md` 索引）、`.agent_state/plans/<task-id>/` 三件套。需要廣泛定位、搜尋或讀大量實作細節時委派 Explore；但當 planner / reviewer 報告影響架構、API contract、測試策略或整合決策時，你應親自讀相關 source / diff / test failure 做 thin-slice sanity check，再決定是否接受報告。

2. **委派優先，但按體量決定是否 self-plan / self-review**。實作、bug 追查、測試、code review 都有專責 agent（見下方委派地圖）。大型或不確定工作預設委派；小型、局部、可在短時間讀懂上下文的工作，可以由 orchestrator 自己產出 implementation plan 或 review finding。你不直接做大量 `Edit`/`Write`；真正屬於統籌的小事（更新 `.agent_state/plans/` 進度、改 roadmap、整理 reports、建立/移除 worktree）才自己動手。

3. **報告是證據，不是結論**。sub-agent 的 final message 與 report 檔是回給你的報告。你讀它們，回寫 `progress.md` / `findings.md`，判斷完成度；但要把報告視為待驗證證據。若報告引用的檔案、測試、風險或設計取捨會左右下一步，至少抽查關鍵 source / diff / test output。報告有矛盾、不足、過度自信或和你抽查結果不一致時，追問、補一輪委派，或自己做更小範圍 review，不要含糊帶過。

4. **不自行猜測或調整架構**（同 `CLAUDE.md`）。架構仍在演化；發現不合理或更好的設計，說明原因交用戶決定，不要擅自改動。實作不確定時同理。

5. **設計先於實作**。用戶提出設計想法時先比較替代方案再委派實作，不要跳過比較直接動手。

## Orchestrator 自驗證與小型工作模式

orchestrator 必須在「委派效率」與「直接理解 codebase」之間取平衡。不要把所有判斷都外包給 sub-agent；也不要把自己變成 implementer。

### 何時親自二次驗證

以下情況應由 orchestrator 親自做 thin-slice verification，再接受 planner / reviewer 結論：

- planner 提議新增/移動 public API、改 module boundary、改資料模型、改 task/worktree protocol。
- reviewer 判定「無問題」，但變更跨模組、測試覆蓋薄、或 diff 觸及核心行為。
- 報告中有模糊語句、未附檔案/line/test evidence、或結論依賴推測。
- implementer 的實作和 planner 的步驟、ADR、模組 README 或用戶要求不完全一致。
- 測試失敗、只跑了部分測試、或驗證環境和目標 workdir 不一致。

二次驗證的尺度要小而精準：讀相關 README/ADR、關鍵 source/diff、測試檔、失敗輸出；必要時跑 targeted `pyright` / `pytest` / `ruff`。若需要廣泛搜尋或長時間分析，再委派 Explore / reviewer。

### 何時自己作為 planner / reviewer

小型工作可以由 orchestrator 直接扮演 planner 或 reviewer，不必機械式委派：

- scope 清楚，通常限於單一模組、少數檔案、文件/skill/protocol 調整，且沒有 live hardware / GUI / external state。
- 你能在當前回合讀完必要 README/ADR/source/diff，並能明確列出測試或驗證方式。
- 風險主要在規則一致性、文件精準度、簡潔性、或小型行為面，而不是大規模實作細節。

self-planner 產出要 source-grounded：引用讀過的高層文件或關鍵檔案，列出可執行步驟、驗證方式與需停下問用戶的分叉。self-reviewer 要採 code-review stance：先列 correctness / regression / missing-test findings；沒有 finding 就說明抽查範圍與剩餘風險。若 review 後需要大量修改，轉回委派或拆 lane，不要硬吃成 orchestrator 工作。

## `.agent_state/` 目錄

`.agent_state/` 是 gitignored agent 工作區，不進 commit。它取代舊 `task_plans/`。

```text
.agent_state/
  plans/
    archives/
      <task-id>/      # completed task plan folders
    <task-id>/
      task_plan.md
      findings.md
      progress.md
      archive.md
  worktrees/
    state.json
    reports/
      <task-id>/
        <lane-id>/
          <agent-id>.md
    trees/
      <worktree-id>/      # git worktree checkout
```

- `.agent_state/plans/<task-id>/task_plan.md`：Goal / Current State / Architecture Baseline / Phase 工作項。
- `.agent_state/plans/<task-id>/findings.md`：委派過程中的非顯而易見發現、決策、踩過的坑。
- `.agent_state/plans/<task-id>/progress.md`：各 Phase / 工作項的狀態與時間軸。
- `.agent_state/plans/archives/<task-id>/`：已結束 task 的完整 plan 目錄；task fast-forward 或明確 abandoned 收尾、且 `state.json` / worktree / branch 都清理完成後，將 `.agent_state/plans/<task-id>/` 整包移到這裡，讓 `.agent_state/plans/` 只保留仍 active / reviewing / merge_preview / blocked 的計劃。這個 `archives/` 是完成 task 的收納區；task 內的 `archive.md` 仍只用於 Phase 壓縮。
- `.agent_state/worktrees/state.json`：仍需操作的 task / lane checkpoint；已 merge / abandoned 並清理完的 task 不留在這裡。
- `.agent_state/worktrees/reports/<task-id>/<lane-id>/`：sub-agent 長報告。報告寫在**主 checkout 的絕對路徑**，不要寫進 task worktree，因為 untracked 檔不會跨 worktree 同步。
- `.agent_state/worktrees/trees/<worktree-id>/`：實際 git worktree checkout；單 lane 預設 `worktree-id = <task-id>`，多 lane 使用 `worktree-id = <task-id>--<lane-id>`。

## Worktree Protocol

本 repo 不使用 taskboard MCP、path lock 或 `agent-taskboard` skill。一般單 agent 工作直接在目前 checkout 完成；需要多 agent、長線 orchestration、或想隔離未提交 diff 時，使用以下 protocol。

### Identity Model

- `task-id` 是 orchestrator session / plan / parent integration branch 的 identity。一個 orchestrator session 負責一個 `task-id`，它錨定 `.agent_state/plans/<task-id>/`、`.agent_state/worktrees/reports/<task-id>/` 與最終要進主線預覽的 `agent/<task-id>`。
- `lane-id` 是同一個 task 內可平行推進的工作線，例如 `impl`、`tests`、`docs`。單 lane task 使用 `main` 作為 `lane-id`，但不需要在 branch / path 名稱中附上 `--main`。
- `worktree-id` 是實際 checkout / lane branch 的 identity。單 lane 時 `worktree-id = <task-id>`、branch 為 `agent/<task-id>`；多 lane 時 `worktree-id = <task-id>--<lane-id>`、branch 為 `agent/<task-id>--<lane-id>`。
- 多 lane task 需要一個 parent integration branch `agent/<task-id>` 作為唯一的主線 preview / final merge 來源；各 lane branch 完成後依序 rebase 到目前的 parent integration branch，再 fast-forward parent integration branch。
- 不要為了「看起來平行」而拆 lane。同一檔案、同一 API contract、同一行為面需要共同設計時，放在同一 lane 序列化；只有 write scope 明確不重疊，或隔離能實質降低整合風險時才開多個 lane。

### State JSON Contract

`.agent_state/worktrees/state.json` 使用這個形狀：

```json
{
  "version": 2,
  "tasks": {
    "<task-id>": {
      "status": "active",
      "base_branch": "main",
      "base_commit": "<sha>",
      "integration_branch": "agent/<task-id>",
      "worktrees": {
        "<lane-id>": {
          "status": "active",
          "role": "lane",
          "branch": "agent/<worktree-id>",
          "worktree_path": ".agent_state/worktrees/trees/<worktree-id>",
          "reports_dir": ".agent_state/worktrees/reports/<task-id>/<lane-id>",
          "write_scope": ["lib/..."],
          "ignored_inputs": []
        }
      }
    }
  }
}
```

`status` 可用：`active`、`reviewing`、`merge_preview`、`blocked`。task-level `status` 是整體狀態；每個 lane 也有自己的 `status`。`role` 可用 `lane` 或 `integration`；單 lane task 通常只有一個 `main` lane，且該 lane branch 同時就是 `integration_branch`。merged / abandoned 的 task 在更新 `progress.md` 並清理 worktree / branch 後直接從 `state.json` 移除。

### 建立 Task / Lane Worktree

1. 選一個穩定 kebab-case `task-id`，例如 `gui-phase-081-session-state`。
2. 先決定 lane 拆分。預設使用單 lane：`lane-id = main`、`worktree-id = <task-id>`。只有需要同一 task 內真正平行修改且 write scope 可切開時，才建立多個 `lane-id`。
3. 以目前目標 base branch/commit 建立 lane branch + worktree：

```bash
git worktree add .agent_state/worktrees/trees/<worktree-id> -b agent/<worktree-id> <base-branch>
```

4. 多 lane task 額外建立 parent integration branch `agent/<task-id>`。需要在獨立 checkout 做整合時，建立 integration worktree，並在 `state.json` 以 `role: "integration"` 記錄；不需要平行整合 checkout 時，至少在 task entry 的 `integration_branch` 記錄它。

```bash
git worktree add .agent_state/worktrees/trees/<task-id>--integration -b agent/<task-id> <base-branch>
```

5. 建立 `.agent_state/worktrees/reports/<task-id>/<lane-id>/`；若這個 task 需要長線計劃檔，建立 `.agent_state/plans/<task-id>/`；並用 `task-id` + `lane-id` + `worktree-id` 更新 `state.json`。
6. 盤點這個 task / lane 需要的額外 gitignored inputs，例如本地設定、scratch fixtures、未追蹤資料檔。worktree 只共享 Git-tracked content；ignored/untracked files 不會自動出現在新 worktree。
7. 對每個額外 gitignored input 選一種處理方式，寫進委派 prompt、report 或 `state.json.worktrees[<lane-id>].ignored_inputs`：
   - **copy**：複製到 task worktree 內的明確路徑，讓 sub-agent 用 worktree-local copy。
   - **reference**：給 sub-agent 主 checkout 的絕對路徑，只讀使用；report 一律用這種方式寫到主 checkout。
   - **omit**：若不需要，明確說不提供，避免 sub-agent 猜測舊 `task_plans/` 或其他本地檔存在。
8. 委派 sub-agent 時明確給：
   - `task-id`、`lane-id`、`worktree-id`
   - lane worktree 的 `workdir`
   - 它擁有的 write scope
   - 主 checkout 的 report 絕對路徑
   - 需要使用的額外 copied/reference inputs
   - 「不要 revert 他人改動；同 task / lane 內可能已有其他 agent 改過」

### 多 Lane 與多 Sub-Agent

一個 task 可以有一個或多個 lane worktree；一個 lane 也可以有多個 sub-agent，但 orchestrator 必須選一種：

- **序列化**：前一個 agent 完成並回報後，下一個才動。
- **不重疊 write scope**：同時派工前先分清檔案/模組責任，避免互相覆蓋。

跨 lane 的 write scope 也要明確不重疊；若兩個 lane 都需要碰同一檔案、同一 public API 或同一測試 fixture，合併到同一 lane 序列化。不要假設 Codex、Claude Code、opencode 或任何 runtime 內建 sub-agent 會自動建立或切換 worktree。需要 worktree 隔離時，由 orchestrator 顯式建立並把 workdir 傳給 agent。

### 驗證環境注意事項

- 如果 task / lane worktree 需要建立自己的 `.venv`，必須用專案的 `development` extra 建立，例如 `uv sync --extra development`（或等價的 `.venv/bin/python -m pip install -e ".[development]"`）。只建立基礎 venv 會缺 `pytest` / `pytest-xdist`，導致收尾驗證無法跑 `pytest`。
- 收尾驗證中的 `pyright` / `ruff` 一律在目標 workdir 用 `uv run` 執行，例如 `uv run pyright`、`uv run ruff check .`、`uv run ruff format .`，避免吃到系統或其它 worktree 的工具版本。

### 整合與線性收尾

1. 收齊 reports，讀懂變更理由、測試結果與風險。
2. 在每個 lane worktree 中檢查 diff，跑必要 pyright / pytest / ruff；確認沒有 sub-agent 還在同一個 `agent/<worktree-id>` branch 上工作後，才可改寫該 lane branch 歷史。
3. lane 完成後，在 lane worktree 中把 `agent/<worktree-id>` rebase 到目前的 parent integration branch（單 lane 時就是 `base_branch` / `agent/<task-id>`），並把內部修補 commit 整理成語意清楚的 lane commit。預設用 `git reset --soft <integration-branch>` 後重新 `git commit`；若保留多個 commit 對審查更清楚，需在 report 說明。
4. 單 lane task：`agent/<task-id>` 同時是 lane branch 與 integration branch，整理後可直接進入主線 preview。
5. 多 lane task：由 orchestrator 決定整合順序。每個 lane branch 依序 rebase 到目前的 `agent/<task-id>`，然後在 integration checkout 中用 `git merge --ff-only agent/<worktree-id>` 推進 parent integration branch。遇到衝突時先判斷是否為規格/架構分歧；若是，停下向用戶確認，不要自行猜測。所有 lane 都進入 `agent/<task-id>` 後，這個 parent branch 才是主線驗收預覽與最終 fast-forward 的來源。
6. 回到主 checkout 的 `base_branch`，由 orchestrator 用 `git merge --no-commit --no-ff agent/<task-id>` 建立未提交的驗收預覽，讓用戶在主線脈絡下檢查整體 diff、測試狀態與行為。這個 merge 只作 preview；不要 commit，也不要在主 checkout 的 merge preview 上直接改碼。
7. 若用戶提出改動意見，先在主 checkout 執行 `git merge --abort` 取消 preview，再委派 sub-agent 回到相關 lane worktree 或 integration worktree 修改、測試、更新 report；完成後重新整理 lane / integration branch，必要時 rebase，再重走 preview。
8. 若用戶沒有改動意見並授權收尾，先取消仍開著的 preview，再用 fast-forward 把 `agent/<task-id>` 接到主線，保持 `base_branch` 線性歷史；然後從 `state.json` 刪除 task entry，結束 Phase，移除 task 的所有 lane / integration worktree 並刪除對應 branch：

```bash
git merge --abort
git merge --ff-only agent/<task-id>
git worktree remove .agent_state/worktrees/trees/<worktree-id>
git worktree prune
git branch -d agent/<worktree-id>
git branch -d agent/<task-id>
```

單 lane 時 `agent/<worktree-id>` 與 `agent/<task-id>` 是同一個 branch，只刪一次。多 lane 時，上述 `worktree remove` / `branch -d` 對每個 lane branch、integration worktree 與 integration branch 各執行一次；若 lane branch 已被 fast-forward 納入 `agent/<task-id>`，`git branch -d agent/<worktree-id>` 應能成功。

若 `.agent_state/plans/<task-id>/` 存在，最後建立 `.agent_state/plans/archives/` 並將整個 plan 目錄移到 `.agent_state/plans/archives/<task-id>/`；不要只複製單一檔案，也不要把已結束 task 留在 active plans 列表旁。

9. 若尚未建立 preview，就直接用 `git merge --ff-only agent/<task-id>` 收尾；不要為了收尾建立 merge commit。
10. 若未授權進入主線 preview，停在可檢查的 integration branch / lane branch diff，回報 task-id、lane-id、worktree path、測試狀態與下一步。
11. 每個 task item 或 Phase 告一段落時，必須完成整合決策：fast-forward、abandon 或 blocked。已結束 task 要同步完成 plan 歸檔；不要把 worktree 或 active plan 當長期常駐狀態留著，長期殘留會讓 branch、ignored inputs、reports 與 base branch 漸漸失同步。
12. 當 task 告一段落且沒有剩餘 backlog 時，可以主動詢問用戶是否關閉這個 task。用戶同意後，依上方收尾規則清理目前 task 的 worktree / branch / `state.json` entry，並將 `.agent_state/plans/<task-id>/` 整包移到 `.agent_state/plans/archives/<task-id>/`。
13. `git merge --squash` 不是預設收尾方式；它會斷開 branch ancestry，讓 `git branch -d` 無法確認 task / lane branch 已整合。除非用戶明確要求 squash merge，否則使用 rebase / soft reset 整理 lane branch，再以 `ff-only` 推進 integration branch 與主線。

### Phase 推薦流水線

需要多 agent、長線 orchestration、或想隔離未提交 diff 的 Phase，預設使用以下流水線。若 Phase 體量小且符合「Orchestrator 自驗證與小型工作模式」，`impl-detail-planner` 或 `python-module-reviewer` 可以由 orchestrator 親自執行，但要在 `progress.md` 或回報中寫明 self-planning / self-review 的範圍、依據與剩餘風險。

1. `impl-detail-planner`：先讀相關 README/ADR/source，把 Phase 目標拆成 source-grounded 實作步驟與測試計劃；只產出報告，不改碼。
2. `plan-item-implementer`：在該 Phase 的 lane worktree 中照 planner 報告逐項實作；遇到架構不明或規格分叉時停下回報，不自行猜測。
3. `python-module-reviewer`：針對 implementer 的變更做 correctness / simplicity / anti-pattern review；review finding 回到同一個 lane worktree 修正，必要時再跑一輪 reviewer。
4. orchestrator：收齊三階段 reports，對 planner / reviewer 的關鍵結論做二次驗證，檢查 diff，更新 `.agent_state/plans/<task-id>/progress.md` / `findings.md`，依 `CLAUDE.md` 收尾驗證。
5. linear preview / 收尾：建立 worktree 時把當前目標 branch 記成 `base_branch`；Phase 變更完成後，先整理各 lane branch，再依序 fast-forward 到 parent integration branch `agent/<task-id>`。回主 checkout 用 `git merge --no-commit --no-ff agent/<task-id>` 建立驗收 preview。用戶有改動意見時 abort preview 並委派 sub-agent 回相關 lane / integration worktree 修改；用戶無意見並授權收尾時 abort preview，改用 `git merge --ff-only agent/<task-id>` 進主線，結束 Phase，移除所有 lane / integration worktree 並刪除 branch。未授權進入 preview 時停在可檢查 diff / branch commit，回報 task-id、lane-id、worktree path 與下一步。

若 Phase 是 bug 診斷或需求仍不明，先插入 `python-bug-investigator` 或 Explore；不要跳過設計/診斷直接實作。

## 委派地圖（task-type → agent）

用 Agent tool 委派；獨立工作項在同一則訊息一次發多個以並行。

| 工作性質 | 委派對象 |
|---|---|
| 廣泛搜尋、定位檔案/符號/命名慣例（要結論不要檔案 dump） | `Explore` |
| 把架構目標拆成 source-grounded 的實作步驟（不寫碼） | `impl-detail-planner`（或內建 `Plan`） |
| 照既定計劃逐項落地成程式碼（不做架構決策） | `plan-item-implementer` |
| bug / 非預期行為的根因診斷（先診斷不亂修） | `python-bug-investigator` |
| 剛寫/改完的模組做正確性 + 簡潔性 + anti-pattern 審查 | `python-module-reviewer` |
| MCP 工具 / skill 文件端到端 dogfooding 與回饋 | `mcp-skill-tester` |
| 跨模塊研究、不確定能否一兩次命中的搜尋、雜項多步任務 | `general-purpose` |

## 進度追蹤模型

一個 orchestrator session 負責（owns）恰好一個 `.agent_state/plans/<task-id>/` 計劃。`task-id` 是 plan / roadmap / findings / parent integration branch 的 identity；`lane-id` 是同一 task 內可平行工作線的 identity；`worktree-id` 是實際 checkout / branch identity。你的 roadmap、進度、findings 都局限在這個 task-id，別去改別的 task plan。唯一共用且會進 commit 的是 `docs/adr/`：跨模組 / 跨 task 的設計決策寫在那裡，以現在式描述目前生效的設計，`[[NNNN]]` 互鏈，先查 `docs/adr/README.md` 索引。

跨模組設計決策寫進 `docs/adr/`，模組局部知識寫進該模組 `README.md`（lib/tests 子目錄）。

## 工作迴圈

1. **釐清目標、定 task-id**。對應到既有 `.agent_state/plans/<task-id>/` 或開新的。目標含糊時用開放式問題向用戶澄清。
2. **建立 context**。讀該 task-id 的三件套 + 相關模組 `README.md`（lib/tests 子目錄）/ ADR。缺對程式碼現狀的理解就委派 Explore。
3. **拆解成 task item / lane**，需要隔離或多 agent 時為每個可獨立整合的 lane 建立 worktree/state entry，並在委派 prompt 中說明額外 gitignored inputs 的 copy/reference policy。
4. **委派 / self-plan / self-review**：每個 Phase 預設走 `impl-detail-planner` → `plan-item-implementer` → `python-module-reviewer`。給每個 agent 清楚 task-id、lane-id、workdir、write scope、report path；planner 產出是 implementer 的輸入，reviewer finding 回到同一 lane worktree 修正。若工作小而局部，可以由 orchestrator 自己完成 planner 或 reviewer 角色，但要明確記錄讀了哪些 source/diff、採納了哪些結論、以及哪些風險仍需委派或用戶決策。
5. **收報告 → 二次驗證 → 整合**：讀 final message 與 report 檔，抽查關鍵 source / diff / test evidence，回寫 `progress.md` / `findings.md`，判斷完成度。報告矛盾、不足或和抽查結果不一致時補一輪委派、要求修正 report，或自己做更小範圍 review，不要含糊整合。
6. **驗證與回報**：一個 Phase 收尾時，按 `CLAUDE.md` 依序跑（或指示/委派）`pyright` → `pytest` → `ruff`，再更新對應模組 `README.md`（lib/tests 子目錄，現在式、刷新頂部 Last updated，不寫 commit hash）。
7. **關閉 task / lane worktree**：Phase / task item 收尾時，先在各 lane branch rebase / 整理 commit，必要時整合到 parent branch `agent/<task-id>`，再用 `git merge --no-commit --no-ff` 在主 checkout 建立驗收預覽；用戶有改動意見就 abort preview 並讓 sub-agent 回相關 lane / integration worktree 修，無意見且授權收尾就 abort preview 後用 `git merge --ff-only agent/<task-id>` 進主線，從 `state.json` 刪除 task entry、移除所有 lane / integration worktree 並刪除 branch。若 task 已告一段落且沒有剩餘 backlog，可以先詢問用戶是否關閉；用戶同意後就按同一套清理流程移除當前 worktree 並歸檔 plan。若不整合，清理後同樣刪除 entry；只有仍在 active/reviewing/merge_preview/blocked 的 task 才保留 entry。task 結束後，將 `.agent_state/plans/<task-id>/` 整包移到 `.agent_state/plans/archives/<task-id>/`，讓 plans 根目錄維持乾淨。

## 邊界

- 不繞過 sub-agent 親手做重活。self-plan / self-review 允許你直接讀必要 source/diff 並形成判斷；不代表可以把大型實作、長時間 bug 追查或全面 code review 都攬下來。發現自己正在大幅 `Edit`/`Write` 實作碼，停下來改成委派或縮小為統籌修改。
- Live singleton 資源（ZCU 板、GUI、固定 port）不靠通用 lock；需要時由 orchestrator 人工序列化，MEASUREMENT 角色仍遵循量測 skill 與 agent-memory。
- Commit / merge：用戶要求或授權才執行。
- 強型別、Fast Fail、責任明確、最小驚訝；不符合即使用戶提出也先警告。除非用戶要求，不保留 legacy / 相容性邏輯。

## 何時不用這個 skill

單一、明確的一次性小修，或純問答 / 查詢，直接做更省事，不必拉起統籌流程與 task worktree。若 orchestrate 已經因長線 task 啟用，而其中某個 item 體量很小，則可使用 self-plan / self-review，不必為了形式再派 sub-agent。
