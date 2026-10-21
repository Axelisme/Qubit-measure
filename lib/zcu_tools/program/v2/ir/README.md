# IR architecture guide

**Last updated:** 2026-10-21

**讀者：** framework-dev。QICK ASM V2 的 IR 分層、分析與 lowering 契約。

本模塊負責把 QICK ASM V2 程式表示成可分析、可優化、可再降階回 assembler 輸入的 IR。

## Core layers and owners

1. `IRLinker` ([linker.py](linker.py))
   - 負責 `list[Instruction]` 與 `prog_list + labels + meta_infos` 的互轉。
   - `link()` 是唯一會寫入 `P_ADDR` 與 `LINE` 的地方。
2. `IRLexer` ([factory.py](factory.py))
   - 負責 `list[Instruction] ↔ list[BasicBlockNode | MetaInst]`。
   - `MetaInst` 在 chunked stream 中獨立於 `BasicBlockNode` 存在。
3. `IRParser` ([factory.py](factory.py))
   - `parse()` 把 `list[BasicBlockNode | MetaInst]` 建構為 `BlockNode` 根節點。
   - `unparse(root: IRNode)` 是唯一的降階入口，接受 tree pass 替換後的任意合法 IR 根節點，輸出含結構 MetaInst 的 chunked stream。原有 `BlockNode` tree 與 `parse()` 保持 round-trip。`IRDispatch` 葉節點由 pass 生成。
4. [dispatch.py](dispatch.py)
   - 提供 loop / branch 共用的 dispatch-table lowering 原語。

[operands.py](operands.py) 擁有 operand 型別與 parse factories。
[instructions.py](instructions.py) 擁有指令基底、compute／control／time 指令與中央 QICK dispatcher。
[io_instructions.py](io_instructions.py) 擁有完整的 memory／port 指令及其 QICK dictionary 互轉。
[instruction_fields.py](instruction_fields.py) 擁有兩組指令共用的有限值契約、QICK 欄位解析與 label reference remap。
[node.py](node.py) 擁有 IR tree 節點、`clone_renamed` 與唯讀共用 interface
`collect_subtree_names`。Clone、parser 與 passes 共用此 interface 收集子樹定義的 label names
及 structure names，名稱配置仍由 [labels.py](labels.py) 擁有。
[labels.py](labels.py) 擁有 label value objects 與名稱配置。
硬體角色性常量集中在 [hw_semantics.py](hw_semantics.py)，避免 magic string 散落各處。
[analysis.py](analysis.py) 提供 cost/size 估算 helper 與 liveness 工具，pass 端優先使用。

## Pipeline

IR 優化是可選層：macro/module 層自身已能生成可運行 ASM，`IRPipeLine` 是純優化器。`disable_all_opt=True` 時 pipeline 原樣回傳輸入。

[pipeline.py](pipeline.py) 的 `IRPipeLine` 使用三種 pass 介面：

- `AbsChunkPass`：ChunkList 層的 per-block 轉換，正確性僅依賴單一 block 內容。
- `AbsChunkListPass`：ChunkList 層的全局轉換，需要看完整 chunk list（如跳轉引用、block 鄰接關係）。
- `AbsIRTreePass`：IR tree 層的 post-order 轉換。
  - `None` → pass 未變動該節點，pipeline 試下一個 pass
  - `IRNode` → 替換子樹，pipeline re-recurse 進新節點（內含的新結構節點也會被處理）
  - 遞歸由 pipeline 統一掌控（post-order）：`transform` 被呼叫時子節點已收斂；pass 若需未展開子樹的資訊（如 unroll 估算 body size），在 `transform` 開頭、回傳前自行分析。

### Pipeline 執行流程

```text
list[Instruction]
  → IRLexer.lex            → ChunkList
  → ChunkList 優化          （ChunkPass + ChunkListPass 同組迭代至收斂/上限）
  → IRParser.parse         → IR tree
  → IR tree 優化            （_optimize_tree：AbsIRTreePass post-order）
  → IRParser.unparse       → ChunkList（含結構 MetaInst）
  → strip 結構 MetaInst     （DISABLE_OPT 資訊已在 BasicBlockNode.disable_opt 屬性）
  → ChunkList 優化          （ChunkPass + ChunkListPass 同組迭代至收斂/上限）
  → IRLexer.flatten        → list[Instruction]
```

[passes/](passes/) 按 `control_flow`、`dataflow`、`timeline`、`loop` 分目錄。

## Invariants that passes rely on

- Operand 的讀/寫語義不在 operand 層決定：所有 `Operand` 子類只實作 `regs() -> frozenset[str]`（中性語義，「這個值涉及哪些暫存器」）。讀還是寫由持有該 operand 的 instruction 自行決定，在 `reg_read` / `reg_write` 屬性中區分。
- `MetaInst` 只能存在於 flat / chunked structural layer，不能進入 `BasicBlockNode.insts`。
- `disable_opt=True` 的兩個來源：（1）dispatch-table stub blocks，pmem word 數與 object identity 不可改變；（2）包含 QICK pseudo-label（`HERE`/`NEXT`/`SKIP`）的 macro 展開，pseudo-label 在 assembler 階段才解析為 `P_ADDR ± offset`，IR pass 若插入、移除或替換 block 會讓 offset 計算錯位。
- `IRLoop.body` 必須視為完整 iteration 的語義單位，不能假設 counter update 一定位於最後一條指令。
- `IRBranch.cases` 保持泛化 `IRNode` contract；analysis helper 必須能估算 direct `BasicBlockNode`、`BlockNode` 與 nested structure case，不能 cast 成 `BlockNode`。
- `IRParser._check_sese()` 會禁止從任一結構 region 外部跳入或透過 label-address instruction 引用該 region 的 internal label，包含 parent region 內部跳入 nested child region 的情況。
- 所有帶 `if_cond` 的 `JumpInst` 不能與 `-op` 寫在同一行。所有的寫法都必須拆分為先 `TEST (op)` 更新 flag，再 `JUMP (if_cond)` 判斷跳轉。
- `ADDR_REG` 是 lowering scratch register，不可作為 dispatch selector 或 branch compare register。`IRBranch.compare_reg`、`IRDispatch.value_reg` 與 high-level `modules.control.Branch(compare_by=...)` 會拒絕 canonical name 為 `s15` 的 register（含 `s_addr` alias），避免 big-PMEM jump setup 覆寫 selector。

## Serialization boundary

`from_dict` 是接收 QICK `prog_list` 的外部序列化邊界。缺少必要 key 時拋出 `KeyError`，parse 失敗時拋出 `ValueError`，讓呼叫端在反序列化時取得明確錯誤。

- IR 中的 `Instruction`（`BaseInst` 的子類，如 `RegWriteInst`, `PortWriteInst`, `WmemWriteInst` 等）的 `from_dict` 與 `to_dict` 實作必須嚴格對應 QICK `tprocv2_assembler.py` 裡的欄位。
- 序列化只使用底層 QICK 支援的欄位，並保留精確欄名，例如 `DST`。`WW`、`WP` 等 side-effects 也必須完整序列化，讓 `prog_list` 符合 QICK 格式。
