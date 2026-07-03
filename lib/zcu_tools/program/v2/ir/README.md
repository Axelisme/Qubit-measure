**Last updated:** 2026-07-03 — optimizer correctness barriers

# IR Architecture Guide

本模塊負責把 QICK ASM V2 程式表示成可分析、可優化、可再降階回 assembler 輸入的 IR。

## Operand Type System (`operands.py`)

`ImmValue` 依語意拆成 4 個獨立型別，各自的 `__str__` 負責輸出對應前綴：

| 型別 | `__str__` | 用途 |
|------|-----------|------|
| `ImmValue(n)` | `n` | 裸整數（port 號、DPORT DATA） |
| `Immediate(n)` | `#n` | ALU 立即值（`REG_WR imm`、`AluExpr.rhs`） |
| `TimeOffset(n)` | `@n` | timed write 的時間偏移欄位 |
| `MemAddr(n)` | `&n` | DMEM/WMEM 的數值記憶體地址 |

`AluExpr.rhs` 型別為 `Optional[Union[Register, Immediate, DmemAddr]]`。`DmemAddr` 是
dmem dispatch table 的**未解析引用**（持 `table_labels: tuple[Label,...]`，不持 offset）：
`DmemDispatchPass` 生成 `REG_WR s15 op (index + DmemAddr)`，pipeline 的 `_resolve_dmem_dispatch`
步驟（所有優化後）才把 `DmemAddr` 換成具體 `Immediate(base)`。`DmemAddr` 是 frozen 引用，
clone 安全；`__str__` raise（resolve 前不可序列化）。詳見 ARCH-6 / Pipeline 章節。

**型別別名：**

- `ValueType = Register | Immediate | ImmValue`
- `AddrType = Register | MemAddr`（不含 `Label`，間接跳轉用 `Register("s15")`）
- `TimeType = Register | TimeOffset`
- `SrcType = SrcKeyword | Register`

**Parse 函數（全為外部 factory，dataclass 本身不含 `parse` classmethod）：**

- `parse_register`、`parse_immediate`、`parse_time_offset`、`parse_mem_addr`、`parse_imm_value`
- `parse_alu_expr`、`parse_side_write`、`parse_label`
- 複合分派：`parse_value`、`parse_addr`、`parse_time`、`parse_src`

**`parse_register` 的 `&` 剝除**：`asm_v2.py` 在 dmem 位址為暫存器時會產生 `'&r1'` 格式（`ReadDmem.expand()` L633：`addr = '&%s' % (prog._get_reg(...))`）。`parse_register` 必須剝除前綴 `&` 才能識別，並回傳裸名的 `Register`（`name='r1'`）。

**`parse_addr` 的解析優先順序**：必須是 `MemAddr（&N）→ Register（含 &rN）`，`addr` 欄位不包含 Label。若傳入 `&name` 格式的 label 字串到 `parse_addr` 會拋出 `ValueError`。

`SrcKeyword` 枚舉（`OP`, `IMM`, `LABEL`, `DMEM`, `WMEM`, `REG`）取代所有字串 src 比較。

`Register` 的語義查詢介面（全部基於 canonical name，alias 透明）：

- `canonical_name: str` property → 解析別名後的單一 canonical 名（`w_freq` → `"w0"`）
- `regs() -> frozenset[str]` → 展開 `r_wave` 為 `{w0..w5}`，否則 `{canonical_name}`；這是 dataflow analysis 的主要入口
- `is_general_reg()` → `r0`、`r1`、... 等通用暫存器。嚴格匹配 `GENERAL_REGS`，不允許描述性名稱（如 `r_cnt`），以確保硬體相容性並及早發現邏輯錯誤。
- `is_wave_reg()` → `w0`~`w5` 或 `r_wave`
- `is_volatile_reg()` → `s0`~`s15`（具硬體副作用的 system reg，包含 `s15` dispatch address reg，排除 `wN`/`r_wave`）

**Operand 的讀/寫語義不在 operand 層決定**：所有 `Operand` 子類只實作 `regs() -> frozenset[str]`（中性語義，「這個值涉及哪些暫存器」）。讀還是寫由持有該 operand 的 instruction 自行決定，在 `reg_read` / `reg_write` 屬性中區分。`reg_read` 和 `reg_write` 均回傳 `frozenset[str]`，不再是 `list[str]`。

## Hardware Constants & Semantic Helpers

硬體角色性常量集中在 `hw_semantics.py`，避免 magic string 散落各處：

- `TIMED_BASE_REG` (`s14`)：所有定時指令的隱性時間基準。
- `USR_TIME_REG` (`s11`)：`WAIT time` / `TIME updt` 讀取。
- `STATUS_REG` (`s10`)：`WAIT port_dt` / `div_rdy` 等讀取。
- `ADDR_REG` (`s15`)：big-PMEM 跳轉與 dispatch base。
- `WAVE_REGS`：`w0..w5` 的 frozenset；`r_wave` bundle 在 `Register.regs()` 以字面值 `"r_wave"` 判斷展開，無獨立常量。
- `VOLATILE_REGS`：`s0..s15`（與 `Register.is_volatile_reg()` 同義的集合形式）。
- `needs_big_jump(pmem_size) -> bool`：純硬體閾值謂詞，`pmem_size > BIG_JUMP_PMEM_THRESHOLD (2**11)` 時回傳 `True`，表示需要 2-word indirect jump idiom（`REG_WR s15 label; JUMP [s15]`）。`dispatch.py`、`factory.py`、`macro/loop.py` 等均從此處 import，不再有重複定義。

`analysis.py` 提供 cost/size 估算 helper 與 liveness 工具，pass 端優先使用：

- `estimate_flat_size`、`estimate_body_scheduled_ticks`、`estimate_body_cost`：用於 unroll 決策。
- `collect_referenced_labels(chunks) -> set[Label]`：收集 ChunkList 中所有被引用的 Label（含 `DmemAddr` 的多重引用），供 `DeadLabelEliminationPass` 與 `BlockMergePass` 使用。函數原位於 `labels.py`，因需 import instruction/node 型別而依賴懶 import；現移至 `analysis.py`（已合法可見這些型別）。`labels.py` 不再有跨層依賴。
- wmem load 判斷（`isinstance(inst, RegWriteInst) and inst.src == SrcKeyword.WMEM`）與 s14 相依判斷（`TIMED_BASE_REG in inst.reg_read`）已 inline 到各自的 pass，不再有獨立 helper。

`instructions.py` 中 `src` 的 keyword/register 判斷邏輯直接 inline 在各指令的 `reg_read` property 內（如 `RegWriteInst.reg_read` 中判斷 `self.src == SrcKeyword.WMEM` / `isinstance(self.src, Register)`），無共用 helper。

## Dataflow Analysis & Hardware Dependency Modeling

為了保證最佳化（如 `IncRegMergePass`, `DeadWriteEliminationPass`）的安全性，IR 必須完整模擬 tProc v2 的硬體行為：

1. **Register Aliasing (暫存器別名)**:
   - `r_wave` 是 `w0`~`w5` 的別名。在 `Register` Operand 中必須實作聯動 ：
     - 寫入 `r_wave` 視為同時寫入 `w0`~`w5`。
     - 讀取 `r_wave` 視為同時讀取 `w0`~`w5`。
     - 寫入任何 `wN` 視為同時修改了 `r_wave` 的狀態。
   - **Symbolic alias normalize 已內聚於 `Register`**: QICK 提供 `w_freq..w_conf`、`s_zero..s_addr` 等別名（見 `operands.py` 的 `_REG_ALIAS`）。`Register.regs()` 對純 alias（如 `w_freq → w0`）只回傳 `{w0}`，pass 端永遠只看 canonical name，不需再呼叫 `canonical_reg()`。`r_wave` 不在 `_REG_ALIAS` 中，是 `w0~w5` 的 bundle reference；`Register("r_wave").regs()` 展開成 `{w0..w5}`，`WmemWriteInst.reg_read` 手動加入 `r_wave` 展開集合，確保 DCE 能感知 wave register 整組讀取。

2. **Implicit Hardware Dependencies (隱性硬體相依)**:
   - **時序相依 (`s14`)**: 所有的定時指令（`WMEM_WR`, `WPORT_WR`, `DPORT_WR`）不論是否有 `@T` 參數，底層皆以 `s14` (out_usr_time) 為基準。因此這些指令必須宣告**讀取 `s14`**，以防止 `delay` (修改 `s14`) 被錯誤地挪動過界。
   - **`@T` 缺省與 `@0` 不可互換**: 「無 `@T`」的 timed write 編碼為 `TO=0`（DATA 留給 `-wr/-op`），「有 `@T`」編碼為 `TO=1`（禁止 `-wr/-op`）。雖然兩者實際發送時間都是 `s14`，但 sink 一個 `TIME inc_ref` 跨越 `time=None` 的 timed write 會改變該指令觀察到的 `s14` → 改變發送時間。`TimedMergePass` 對 `reg_read` 含 `s14` 的指令必須 flush pending TIME 而僅僅是 fold `@T`。
   - **`TimedMergePass` 的溢出保護**: `TIME inc_ref #N` 的 LIT 欄位與 `@T` 欄位皆為 32-bit signed（硬體上限 `2^31-1`）。`TimedMergePass` 採用保 守安全值 `TIMED_LIT_MAX = 2^20 - 1`。兩個檢查點：(1) `pending_lit + delta > TIMED_LIT_MAX` 時先 flush，若 `delta` 本身也超過上限則直接輸出不累加；(2) `time.value + pending_lit > TIMED_LIT_MAX` 時 flush pending 並保留原 `@T` 不調整。溢出時保守 flush，語義不變，只放棄合併。**`pending_lit` 在 `@T` 吸收後不清零**：後續同一 baseline segment 的 timed inst 也需加上同樣 delta，且 block 末尾必須 emit 一個 TIME inst 讓硬體 reference clock 實際推進。
   - **`TimedMergePass` 的 s14 read/write 屏障**：`pending TIME inc_ref` 不可越過任何**非 `TimeInst`** 且讀取或寫入 `s14` 的指令，因為那類指令會觀察或修改時間基準，使已累積的 delta 不能被 sink 到其後。顯式 `TimeOffset(@T)` fold 分支先處理，允許 anchored timed instruction 吸收 pending delta；其餘非 `TimeInst` 以 `TIMED_BASE_REG in inst.reg_read or TIMED_BASE_REG in inst.reg_write` 作為 flush 條件。
   - **`IncRegMergePass` 的溢出保護**: `REG_WR rd op (rs +/- #N)` 的立即數欄位為 24-bit signed（硬體上限 `2^23-1`）。`IncRegMergePass` 採用保守 安全值 `INC_REG_IMM_MAX = 2^20 - 1`。累加前先檢查 `abs(new_total) > INC_REG_IMM_MAX`；若溢出則 flush 舊累積、若單步本身就超限則直接 emit as-is  繞過 pending。
   - **計時器讀取 (`s11`)**: `WAIT` 與 `TIME updt` 會讀取 `s11` (curr_usr_time)。
   - **`WAIT` 相依性分類**:
     - `WAIT time`：assembler 展開為 `TEST (s11 - #(TIME - OFFSET))`，實際只讀 `s11`（`TIME` 是已算好的立即數）。`WaitInst.reg_read` 仍**保守宣告同時讀 `s11` 與 `s14`**——這是 over-approximation，用來阻止 pass 把修改 `s14` 的指令挪過 `WAIT`。注意 `TimedMergePass` 另外把 `WaitInst` 當 flush barrier 顯式處理，不依賴此 s14 宣告。
     - `WAIT port_dt/div_rdy/div_dt/qpa_*`：展開為 `TEST (s10 AND #mask)`，僅讀取 `s10`，不讀 `s11` 或 `s14`。

3. **Volatile Registers & Side Effects (易失性寄存器與副作用)**:
   - **`s0`-`s14` 保護**: 這些寄存器通常具備硬體副作用（如 `s1` 隨機數、`s12` 外部進度更新）。`DeadWriteElimination` 絕對不可刪除對這些寄存器的 寫入，即使後續看似無讀取。
   - **SideWrite (`-wr`) dataflow**: 支援 `wr: SideWrite | None` 的 instruction 必須把 `wr.regs()` 納入 `reg_write`。`DeadWriteEliminationPass` 仍只把純 register-like `RegWriteInst` 當刪除候選；memory/port/timing/control instruction 即使 side-write register 之後被 shadow，也不可因此被刪除。
   - **Flag 更新 (`-uf`)**: 帶有 `uf` 標記的指令具備隱性副作用（更新 ALU Flag），不可被 DCE 刪除。
   - **系統寄存器屏障**: `IncRegMergePass`應禁止合併或挪動任何 canonical 名以 `s` 開頭的寄存器增量（包含別名解析後落在 `s` 的）；這保證 I/O 操 作（如 Shot counter）的物理位置穩定性。
   - **wmem 讀作為 read barrier**: `REG_WR <dst> wmem` 從 wave memory 讀取（具有非寄存器副作用）並寫入整組 wave register。DCE 必須把它視為 opaque read barrier：清空 pending 並不把 writes 加進 pending，避免後續 `wN`  寫被當成 shadow 而誤刪這條 wmem 讀。

4. **DCE 對 alias 寫的保守策略**:
   - `RegWriteInst.reg_write` 對 `dst=r_wave`/`wN` 會展開出多個 reg name（`len(writes) > 1`）。為避免複雜的 group shadow 分析，DCE 把這類寫的舊 shadow 加進 dead 集合後，**不**把這次寫放進 pending（即連續多個 wave reg 寫不會互相 shadow）。這是已知的保守，安全但 sub-optimal；若未來要強化，需要設計 group-aware shadow tracking。

## Core Layers

1. **`IRLinker`** (`ir/linker.py`)
   - 負責 `list[Instruction]` 與 `prog_list + labels + meta_infos` 的互 轉。
   - `link()` 是唯一會寫入 `P_ADDR` 與 `LINE` 的地方。
   - `link()` 遇到重複 `LabelInst.name` 會立即 `raise ValueError`，避免 labels dict 靜默覆蓋先前定義。
   - `unlink()` 依 `meta_infos` 重建 `LabelInst` 與 `MetaInst`，回傳 `list[Instruction]`。

2. **`IRLexer`** (`ir/factory.py`)
   - 負責 `list[Instruction] ↔ list[BasicBlockNode | MetaInst]`。
   - `LabelInst` 會開新 block，`JumpInst` 會結束 block。
   - `MetaInst` 永遠作為 chunked stream 的獨立元素存在，不 屬於任何 `BasicBlockNode`。

3. **`IRParser`** (`ir/factory.py`)
   - 負責 `list[BasicBlockNode | MetaInst] ↔ BlockNode`。
   - `parse()` 回傳 `BlockNode`（已無 `RootNode`）；把成對的結構標記還原成 `IRLoop` / `IRBranch`。
   - `unparse()` 是**唯一**的降階入口：把 IR tree 降階回 chunked stream（含結構 MetaInst），與 `parse()` 形成 round-trip 反函數（`IRDispatch` 例外——它是 pass 才生成的葉節點，不會被 `parse` 重建）。
   - `_unparse_node(node: IRNode)` 是核心展開方法，遞迴把任意 `IRNode` 降階到 `list[BasicBlockNode | MetaInst]`。`IRLoop` 降階為 `LOOP_START + 由 _loop_prologue 產生的 prologue + LOOP_BODY_START + 遞迴展開的 body + LOOP_BODY_END + _loop_epilogue + LOOP_END`。
   - `_loop_prologue` / `_loop_epilogue` 只回傳 loop 的 pre / post `BasicBlockNode`（不含 body）；兩者共用同一組 `start`/`end` `Label`，由 `_unparse_node` 一次性 allocate 後傳入。
   - `IRBranch` 的 fallback lowering 會插入 synthetic end-of-case jump 與 trailing `end_label` block；`parse()` 讀回這種形狀時會把這些 fallback skeleton 剝除，因此 `parse(unparse(IRBranch))` 仍回到「純 case body」AST，而不把 skeleton 汙染進 `IRBranch.cases`。
   - IRParser 不再有「無 meta」的展平路徑；structure 樹的展平統一靠 `unparse`。`clone_renamed`（`node.py`）負責複製子樹用於 loop unroll。

4. **`dispatch.py`** (`ir/dispatch.py`)
   - 提供 loop / branch 共用的 dispatch-table lowering 原語。
   - `emit_dispatch_address_setup()` 會算出 `s15 = &table_0 + index * entry_words`。
   - `build_dispatch_table_island()` 只建立固定寬度的 table stub blocks 。

## Node Model

- `Instruction` 是所有 IR instruction 的抽象基類。
- `BaseInst` 是所有真實機器指令的基類，具備 `addr_inc`、`reg_read: frozenset[str]`、`reg_write: frozenset[str]`、`need_label`、`need_labels`、`remap_labels`。`need_label` 回傳單一 `Optional[Label]`；`need_labels -> frozenset[Label]` 涵蓋**所有**被引用的 label（dead-label 分析用），預設由 `need_label` 推導，`RegWriteInst` override 以納入 `DmemAddr.table_labels`（dmem dispatch table 引用的多個 entry label）。`remap_labels(remap: dict[str, Label]) -> BaseInst` 預設回傳 self（identity）；`JumpInst`、`RegWriteInst`、`DmemReadInst`、`CallInst` override 以更新其 `LabelRef`；這讓 `node.py` 的 `_clone_node` 不需要枚舉各指令型別，直接呼叫 `inst.remap_labels(label_remap)` 即可。
- `LabelInst` 與 `MetaInst` 不繼承 `BaseInst`，不佔用 program memory。
- `IRNode` 是所有 IR 樹節點的抽象基類，定義統一的 `children() -> list[IRNode]` 與 `replace_child(old, new)` 介面。葉節點回傳 `[]` 並 raise `TypeError`。
- `BasicBlockNode` 是葉節點，表示 straight-line code：
  - `labels` 只放 `LabelInst`
  - `insts` 只放 `BaseInst`
  - `branch` 只放 terminal `JumpInst`
  - `insts` 中禁止 `MetaInst`、`LabelInst`、`JumpInst`
- `BlockNode` 是結構容器，子節點可以是任何 `IRNode`（`BasicBlockNode`、`IRLoop`、`IRBranch`、`IRDispatch`、`BlockNode`）。`parse()` 回傳 `BlockNode` 作為 AST 根節點（無 `RootNode`）。
- `IRLoop.body: IRNode`（不限定 `BlockNode`），代表完整的一次 iteration，包含 loop-carried counter update。
- `IRBranch.cases: list[IRNode]`（不限定 `BlockNode`），只承載各 case 的實際執行內容，不包含 dispatch tree。
- `IRDispatch` 是葉節點，持有 `value_reg: Register` 和 `target_labels: list[Label]`。`target_labels[-1]` 永遠是 out-of-range guard 的跳轉目標。Case bodies 不放在 `IRDispatch` 內，由呼叫端負責在 chunk stream 中跟在 dispatch 結構後面。
- `disable_opt` **只存在於 `BasicBlockNode`**。其他 IRNode 都不持有這個屬性。
- `clone_renamed(node, allocated)`（`node.py`）深拷貝一棵子樹，並把所有內部 `LabelInst` 名稱與巢狀 `IRLoop`/`IRBranch`/`IRDispatch` 的 `name` 都換成在 `allocated` 中唯一的新名。loop unroll 複製 body k 份時用它確保各份 label 不衝突——結構 `name` 也要換，因為 `unparse` 會用 `name` 合成 `{name}_start` 等 label。

## Structural Meta Format

- Loop 使用以下頂層 `MetaInst`：
  - `LOOP_START`
  - `LOOP_BODY_START`
  - `LOOP_BODY_END`
  - `LOOP_END`
- Branch 使用以下頂層 `MetaInst`：
  - `BRANCH_START`
  - `BRANCH_CASE_START`
  - `BRANCH_CASE_END`
  - `BRANCH_END`
- Dispatch 使用以下頂層 `MetaInst`（由 `_lower_dispatch` 在 unparse 時產生，不出現在 IR tree）：
  - `DISPATCH_START`
  - `DISPATCH_END`
- `BRANCH_START.info["compare_reg"]` 是 parser 還原 `IRBranch` 的必要欄位。
- `BRANCH_CASE_START/END` 包住單一 case 的 body；dispatch blocks（`DISPATCH_START/END` 及其內容）位於 case 外部，在 `BRANCH_START` 後緊跟。
- `DISABLE_OPT_START` / `DISABLE_OPT_END`：包住任何包含 QICK pseudo-label 的 macro 展開範圍，讓 `IRLexer` 把夾住的 `BasicBlockNode` 標記為 `disable_opt=True`。目前由 `AdditionalMacroMixin` 的 `wait()`、`wait_auto()`、`end()` 插入。

## Lowering Shapes

- `IRLoop` lowering 採 do-while 形狀：
  - 動態 `n` 先做 `n == 0` guard
  - `REG_WR counter imm #0`
  - `start` label
  - body
  - cond back-edge `TEST (counter - n)` + `JUMP start -if(...)`
  - `end` label
  - **所有帶 `if_cond` 的 `JumpInst` 不能與 `-op` 寫在同一行**。tProc v2 的 JUMP 指令在帶有 `UF=1` 且同時包含 `if_cond` 和 `op` 時，會平行評估：`IF` 條件測試的是**執行前**（即上一條 `UF=1` 指令留下的）stale flag，同時將新的 `op` 結果寫入下一週期的 flags。因此，若將 `TEST` 與跳轉寫成一條 `JUMP -if(S) -op(...) -uf`，迴圈 back-edge 會吃到迴圈啟動前的 stale flag（通常是 S=0），導致條件永遠不成立，迴圈只跑一次。這適用於所有生成條件 JUMP 的地方：`_loop_prologue`/`_loop_epilogue`、`_lower_dispatch`（factory.py）、`UnrollLoopPass`（unroll.py）、`SimplifyDispatchPass`、`DmemDispatchPass`、以及 macro 層 的 `_emit_cond_jump`。所有的寫法都必須拆分為先 `TEST (op)` 更新 flag，再 `JUMP (if_cond)` 判斷跳轉。
- `IRBranch` 是「被 pass 展開的結構節點」（與 `IRLoop` 對稱）：
  - **正常路徑**：`UnpackIRBranchPass`（IRTreePass）在 tree 階段把 `IRBranch` 展開成
    `BlockNode([IRDispatch, case_0, case_1, ..., end_label_BB])`。每個非最後 case 末尾插
    無條件跳轉到 `end_label`（防 fall-through 進下一 case）。`UnpackIRBranchPass` 會優先沿用
    case body 已有的 entry label（macro 層 emit 的 `{name}_case_entry_i`）；若 case 沒有 head
    label，則從 pipeline 共享的名稱集合配置新 label，保證同一棵 tree 中多個同名 branch 仍全域唯一。
  - 展開出的 `IRDispatch` 由 pipeline re-recurse → `SimplifyDispatchPass`/`DmemDispatchPass` 處理。
  - **fallback 路徑**：`disable_all_opt` 或測試直接 unparse `IRBranch` 時，走 `_unparse_node`
    的 `IRBranch` 分支（emit `BRANCH_START/CASE_START/CASE_END/BRANCH_END` + `_lower_dispatch`
    的 pmem island）。這條路徑只是一個 lowering 形狀；`parse()` 會把它清理回純 `IRBranch` AST。
- `IRDispatch` 的降階由 pass 決定（葉節點，case bodies 是它的 sibling，不在它內部）：
  - **k==2**：`SimplifyDispatchPass` → `BlockNode([cond_bb, fallthrough_bb])`：`cond_bb` 發出
    條件跳轉到 `target_labels[1]`（NZ），`fallthrough_bb` 發出無條件跳轉到 `target_labels[0]`。
    small-PMEM 時 `BranchEliminationPass` 可消除 `fallthrough_bb` 的 unconditional jump（若
    `target_labels[0]` 緊接）；big-PMEM 時兩個 block 均用 `REG_WR s15 label; JUMP [s15]` idiom，
    無 dispatch table。兩個 branch 均顯式 jump，不依賴 fall-through。
  - **k>=2**：`DmemDispatchPass` → `BlockNode`：guard（out-of-range 跳 `target_labels[-1]`）
    - `REG_WR s15 op(index + DmemAddr)` + `REG_WR s15 dmem[s15]` + `JUMP [s15]`。1 跳，
    dmem table，無 `disable_opt`。兩個 pass 都能處理 k==2；通常 `SimplifyDispatchPass`
    先運行，但若 `SimplifyDispatchPass` 不在 pass chain 中，`DmemDispatchPass` 也能正確
    處理 k==2（產生 1-entry dmem table + guard）。k==1 直接 skip（無意義的 dispatch）。
  - **fallback**：未被 pass 轉換的 `IRDispatch` 走 `_lower_dispatch` 的 pmem island
    （`DISPATCH_START` + guard + prologue + 固定寬度 stubs（`disable_opt=True`）+ `DISPATCH_END`）。
- register-driven loop unroll（`_maybe_build_jump_table`）回傳 `BlockNode`，內含：
  - remainder / offset 計算（prologue BasicBlockNodes）
  - counter reset（獨立 BasicBlockNode，在 dispatch 之前）
  - `IRDispatch(value_reg=i, target_labels=entry_labels)`（葉節點，由 pipeline 遞歸 lower）
  - k 個 body `BlockNode`（各自帶 entry label BB，由 pipeline 遞歸 lower）
  - back-edge：`TEST (i - n)` + `JUMP entry_0 -if(S)`，exit label 為 fall-through

## Invariants That Passes Rely On

- `MetaInst` 只能存在於 flat / chunked structural layer，不能進入 `BasicBlockNode.insts`。
- `disable_opt=True` 的兩個來源：（1）dispatch-table stub blocks——pmem word 數不可改變；（2）包含 QICK pseudo-label（`HERE`/`NEXT`/`SKIP`）的 macro 展開——pseudo-label 在 assembler 階段才解析為 `P_ADDR ± offset`，IR pass 若插入或移除指令會讓 offset 計算錯位。
- `IRLoop.body` 必須視為完整 iteration 的語義單位，不能假設 counter update 一定位於最後一條指令。
- `IRParser._check_sese()` 會禁止從結構外部跳入 loop / branch 的控制 label。

## Pipeline

IR 優化是**可選**層：macro/module 層自身已能生成可運行 ASM，`IRPipeLine` 是純優化器。`disable_all_opt=True` 時 pipeline 原樣回傳輸入。

`IRPipeLine` 使用三種 pass 介面：

- `AbsChunkPass`：ChunkList 層的 per-block 轉換，正確性僅依賴單一 block 內容。
- `AbsChunkListPass`：ChunkList 層的全局轉換，需要看完整 chunk list（如跳轉引用、block 鄰接關係）。
- `AbsIRTreePass`：IR tree 層的 post-order 轉換。`transform(node, ctx) -> Optional[IRNode]`：
  - `None` → pass 未變動該節點，pipeline 試下一個 pass
  - `IRNode` → 替換子樹，pipeline re-recurse 進新節點（內含的新結構節點也會被處理）
  - 遞歸由 **pipeline** 統一掌控（post-order）：`transform` 被呼叫時子節點已收斂；pass 若需未展開子樹的資訊（如 unroll 估算 body size），在 `transform` 開頭、回傳前自行分析。

### Pipeline 執行流程（U-shape 單次優化）

```
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

- `_optimize_tree`：post-order 遞迴；先把子節點轉換至收斂，再對父節點跑 `AbsIRTreePass` chain。pass 回傳新節點就 re-recurse。**不**在此跑 ChunkPass。
- `_run_chunklist_opt`：ChunkPass 與 ChunkListPass 放**同一組**迭代（互相可觸發對方的工作）。達 `max_opt_iterations` 仍未收斂時記 `logger.warning`，**並列出仍回報 changed 的 pass 名稱**——這是震盪訊號，非正確性問題（結果仍是合法程式，只是未達最佳）。
- strip 結構 MetaInst（`_strip_structural_meta`）移除 `LOOP_*`/`BRANCH_*`/`DISPATCH_*`/`DISABLE_OPT_*`；`disable_opt` 已隨 `BasicBlockNode` 屬性走，不需保留 meta。第二輪 ChunkList 優化因此能跨原結構邊界做 block merge 等。

### Passes 目錄結構

```
passes/
├── base.py               # DATAFLOW_TRANSPARENT_INSTS, BlockChunkPass
├── control_flow/
│   ├── dead_label.py          # DeadLabelEliminationPass (AbsChunkListPass)
│   ├── branch_elimination.py  # BranchEliminationPass (AbsChunkListPass)
│   ├── block_merge.py         # BlockMergePass (AbsChunkListPass)
│   ├── unpack_branch.py       # UnpackIRBranchPass (AbsIRTreePass)
│   ├── simplify_dispatch.py   # SimplifyDispatchPass (AbsIRTreePass)
│   ├── dmem_dispatch.py       # DmemDispatchPass (AbsIRTreePass)
│   └── unreachable.py         # UnreachableEliminationPass (AbsChunkListPass)
├── dataflow/
│   ├── dead_write.py     # DeadWriteEliminationPass (BlockChunkPass)
│   ├── dead_test.py      # DeadTestEliminationPass (BlockChunkPass)
│   └── inc_reg_merge.py  # IncRegMergePass (BlockChunkPass)
├── timeline/
│   ├── zero_delay_dce.py # ZeroDelayDCEPass (BlockChunkPass)
│   └── timed_merge.py    # TimedMergePass (BlockChunkPass)
└── loop/
    └── unroll.py         # UnrollLoopPass (AbsIRTreePass)
```

IR-tree pass chain 順序：`UnrollLoopPass → UnpackIRBranchPass → SimplifyDispatchPass
→ DmemDispatchPass`。`UnpackIRBranchPass` 把 `IRBranch` 展開成含 `IRDispatch` 節點的
`BlockNode`（使 branch 的 dispatch 成為 tree node）；`SimplifyDispatchPass` 處理
`IRDispatch(k==2)` 成單一條件跳轉、`DmemDispatchPass` 處理 `IRDispatch(k>=2)` 成 dmem
table（k==2 時 `SimplifyDispatchPass` 先到所以 `DmemDispatchPass` 通常看不到 k==2）。
`_unparse_node` 的 `IRBranch` 分支與 `_lower_dispatch` 保留作 fallback（`disable_all_opt`、
測試直接 unparse）。

**Pass 基類設計：**

- `BlockChunkPass(AbsChunkPass)` — 抽象基類，自動迭代 chunks 中的 `BasicBlockNode`，子類只需實作 `_process_block(block) -> bool`。
- `AbsChunkListPass` — 需要看完整 chunk list 的 pass 基類（`DeadLabelEliminationPass`、`BranchEliminationPass`、`BlockMergePass`、`UnreachableEliminationPass`）。
- `AbsIRTreePass` — post-order tree pass，`transform(node, ctx)` 回傳 `IRNode`（同一物件=不動，新物件=re-recurse）。`UnrollLoopPass` 從 IR tree 估算 body size；`SimplifyDispatchPass` 把 `IRDispatch(k==2)` 換成 `BlockNode([cond_bb, fallthrough_bb])`，兩個 branch 均顯式 jump，不依賴 fall-through。
- `DATAFLOW_TRANSPARENT_INSTS` — 17 個對 dataflow tracking 透明的 instruction 型別 tuple，`DeadWriteEliminationPass` 使用它判斷何時清空 pending tracking。`CallInst` / `RetInst` 是 subroutine boundary，不屬於透明 instruction。
- `collect_referenced_labels(chunks)` 是 `analysis.py` 提供的共用 utility；`DeadLabelEliminationPass` 和 `BlockMergePass` 共用同一份實作，避免 label analysis 依賴 instruction/node 型別造成 import cycle。它走 `BaseInst.need_labels`，因此 dmem dispatch table 透過 `DmemAddr` 引用的 entry label 也會被視為「存活」，不被 dead-label 刪除。
- `UnrollLoopPass`（`AbsIRTreePass`）：`transform` 對 `IRLoop` 分派至 `_unroll_full` / `_unroll_partial` / `_maybe_build_jump_table` 三條路徑。register-driven path 回傳 `BlockNode`（含 prologue BasicBlockNodes + `IRDispatch` 節點 + k 個 body `BlockNode` + back-edge）；`IRDispatch` 節點由 pipeline 遞歸處理，讓 `SimplifyDispatchPass` 可在 k==2 時介入。
- `SimplifyDispatchPass`（`AbsIRTreePass`）：`transform` 在 `IRDispatch(k==2)` 時回傳 `BlockNode([cond_bb, fallthrough_bb])`——`cond_bb` 發條件跳轉到 `target_labels[1]`（NZ），`fallthrough_bb` 發無條件跳轉到 `target_labels[0]`，消除 dispatch island；k!=2 回傳 `None`（不變動），讓 `DmemDispatchPass` 或 `_lower_dispatch` 處理。
- `UnrollAnalysis`（dataclass）：`scheduled_ticks: int`、`slack: int` 皆為非 Optional — `estimate_body_scheduled_ticks` 永遠回傳 `int`，不存在 None 狀態。

頂層 `passes/__init__.py` 維持公開 export，外部 import 路徑不受影響。

- `disable_opt=True` blocks：不可改變 block 內的指令與 program-memory word 數，pipeline 有 runtime 驗證。
- `make_default_pipeline(pmem_capacity)` 的參數**一定覆蓋** `DEFAULT_PIPELINE_CONFIG.pmem_capacity`；`PipeLineConfig.pmem_capacity` 是 `int` 型別，不存在 `None` 狀態。
- `PipeLineContext` 持有 `config`、`pmem_budget`、`available_regs`、`allocated_names`。`allocated_names`
  是**單次 pipeline run** 內共享的名稱集合，從 parse 後整棵 tree 的 label / structure name 預先蒐集，
  供 tree-pass（目前主要是 `UnpackIRBranchPass`）配置全域唯一 label；它不是跨 run 的全域狀態。
- `UnrollLoopPass` 展開後回傳的 `BlockNode` 不含 `IRLoop` wrapper（partial unroll 用裸 label + back-edge `JUMP` 而非 `IRLoop`），因此不會被 `_optimize_tree` 重新 unroll；這是防止 pipeline 反覆展開同一個 loop 的機制。巢狀且未被展開的 `IRLoop` 仍以 `IRLoop` 形式保留在 cloned body 中，由 pipeline re-recurse 後正常 lower。
- 結構→ChunkList 的降階統一由 `IRParser.unparse` 負責；不存在「無 meta」的 lowering 路徑。`_optimize_tree` 只跑 `AbsIRTreePass`，不做 lowering。

## Labels and Addresses

- `Label` 是 frozen dataclass value object，以 `name: str` 為基礎做 `__eq__`/`__hash__`，沒有全域 singleton 狀態。`Label.__str__` 回傳**裸名**（`name`，不帶 `&` 前綴）；`&` 是 QICK 序列化格式，屬於呼叫端關注（`linker.py` 的物理地址 `f"&{p_addr}"` 與 `Label.__str__` 無關）。
- `PseudoLabel = Literal["PREV", "HERE", "NEXT", "SKIP"]`：hardware pseudo label，以 str 表示。
- `LabelRef` 是獨立的 frozen dataclass，持有 `target: Union[Label, PseudoLabel]`；提供 `is_pseudo() -> bool` 和 `as_label() -> Label` 方法。`str(LabelRef)` 的語義等於 QICK LABEL 欄位的序列化形式：普通 label → `{name}`，pseudo → `"HERE"` 等。所有跳轉目標（`JumpInst.label`、`RegWriteInst.label` 等）的型別皆為 `Optional[LabelRef]`，不再是裸 `Label`。
- `LabelInst.name: Label`：定義點仍用 `Label`，不用 `LabelRef`。
- `AddrType = Register | MemAddr`：addr 欄位**不包含** Label；間接跳轉統一使用 `Register("s15")`。
- `make_label(base, allocated)` (`labels.py`)：在 `allocated: set[str]` 中找不衝突的名字（加數字後綴），把名字加入 `allocated` 後回傳 `Label`。
- **`allocated` 的生命週期依用途分兩層**：`clone_renamed()` 接收呼叫端提供的 `allocated` set，並把新名字寫回該 set，讓 successive clones 不碰撞；`UnrollLoopPass` 使用 `PipeLineContext.allocated_names` 作為單次 pipeline run 的 shared name pool，並先把 loop body 既有 label / structure name 加入同一 pool。IRParser 的 `allocated` 仍是 per-instance local set，不跨 pipeline run 共用；direct `unparse()` 會先 seed 目前 subtree 的既有 label / structure name，再配置 fallback synthetic labels。
- 當 `pmem_size > 2**11` 時，label jump 需走 big-jump 形式：先把 label 寫入 `s15`，再 `JUMP ADDR=s15`。
- dispatch-table entry width 也因此分成兩種：
  - small PMEM: `1` word (`JUMP label`)
  - big PMEM: `2` words (`REG_WR s15 label target` + `JUMP s15`)
- `emit_dispatch_address_setup()` 在 big PMEM 不會改動語義 index register，而是對 `s15` 做兩次 `+ index`，得到 `base + 2*index`。

## `from_dict` Parse Helpers (`instructions.py`)

`from_dict` 是外部序列化邊界（接收 QICK `prog_list`），不使用任何防禦性 降級（`d.get("KEY", "")` 等 fallback）。缺少必要 key 直接 `KeyError`，parse 失敗直接 `ValueError`，讓 bug 在反序列化時立即暴露。

QICK key 的存在性規則（來自 `tprocv2_assembler.py` 分析）：

- `TIME.C_OP`、`WAIT.C_OP`：assembler 永遠設置，必要。
- `WMEM_WR` 的地址：assembler 只用 `DST`（parse 時 `ADDR` 被轉成 `DST`），不存在 `ADDR` fallback。
- `MetaInst`：只有 `kind/type/name/info` 格式，`CMD/__META__` 舊格式已移除。

針對欄位型別比通用 parse 函數更窄的情況，定義了語意明確的 helper（有特殊 raise 訊息者保留，簡單邏輯直接 inline）：

| Helper | 回傳型別 | 用途 |
|--------|----------|------|
| `_require_register(val, field)` | `Register` | `REG_WR` / `DMEM_RD` 的 `DST` |
| `_require_alu_expr(val, field)` | `AluExpr` | `TEST` 的 `OP` |
| `_parse_port_dst(val)` | `Register \| ImmValue` | `WPORT_WR` / `DPORT_WR` 的 `DST`（port 號是裸整數） |
| `_parse_mem_addr_field(val, field)` | `Register \| MemAddr` | `WPORT_WR.ADDR`、`WMEM_WR.DST`、`DMEM_WR.DST` |
| `_parse_dmem_src_keyword(val)` | `Literal["imm", "op"]` | `DMEM_WR.SRC`（QICK 用 keyword `imm`/`op` 判斷來源，不是值型別） |

不使用 `cast()` 壓過型別錯誤——所有窄化都應反映在真實的 parse 邏輯中。

**`DPORT_WR.DATA` strict parsing**：`DportWriteInst.from_dict()` 不走 `parse_value()` 的寬鬆 fallback，改用專用 helper 僅接受合法 `Register` / `Immediate` / `ImmValue`。像 `DATA="garbage"` 或 `DATA="w_bogus"` 這類值必須在 IR parse 階段直接 `raise ValueError`，不能延後到 assembler。

**`DmemWriteInst` 的 `DST` bracket 處理**：QICK 的 `prog_list` 中 `DMEM_WR.DST` 是 `"[&N]"` 格式（含方括號，來自 `WriteDmem.expand()`）。`from_dict` 先剝除 `[` `]` 再交給 `_parse_mem_addr_field`，`to_dict` 對 `MemAddr` 型別包回 `[&N]` 格式。型別層級定義了 `DmemSrc = Literal["imm", "op"]` 型別別名。

**TimedMergePass set_ref/updt/rst flush**：`TIME set_ref`/`updt`/`rst` 將 `s14` 設為絕對值，使 pending literal delta 失效。這些 `c_op` 在 inc_ref 處理之後有獨立分支觸發 flush，與 inc_ref 區分開來。

**DeadWriteEliminationPass if_cond 處理**：帶 `if_cond` 的 instruction 視為 barrier（清除 pending tracking），因為條件執行的寫入不可靠地 shadow 前一條無條件寫入。

**DeadTestEliminationPass flag liveness**：任何帶 `if_cond` 的 instruction 都會消耗目前 ALU flags，不限於 `JumpInst`。若同一 instruction 同時帶 `if_cond` 與 `uf`，硬體語意是先用舊 flag 判斷條件，再寫入新 flag；因此前一條 pending `TEST` 是 live。CALL/RET 等 opaque boundary 會清掉 pending `TEST` tracking，但不把它標成 dead，因為 boundary 另一側可能觀察 flags。

**完整指令集覆蓋**：所有 QICK tProc v2 合法指令皆有對應的 IR Instruction 類別（`DPORT_RD`, `TRIG`, `CALL`, `RET`, `FLAG`, `ARITH`, `DIV`, `NET`, `COM`, `PA`, `PB`, `CLEAR`）。每個新類別都實作 `from_dict`/`to_dict`/`reg_read`/`reg_write`。dataflow passes 的白名單已同步擴充。

**`c_op` 和 `src` 強型別化**：所有指令的 `c_op`、`src`、`cmd` 欄位皆已換成 `Literal[...]` 型別，對應 assembler 的合法值集合。`from_dict` 一律呼叫 `_require_literal(val, field, valid_set)` 做 runtime 驗證，值不合法立即 `raise ValueError`，錯誤訊息包含欄位名稱與所有合法值。完整的 Literal 型別別名：

| 型別 | 合法值 |
|------|--------|
| `TimeCOp` | `"inc_ref"`, `"set_ref"`, `"updt"`, `"rst"` |
| `FlagCOp` | `"set"`, `"clr"`, `"inv"` |
| `WaitCOp` | `"time"`, `"port_dt"`, `"div_rdy"`, `"div_dt"`, `"qpa_rdy"`, `"qpa_dt"` |
| `ClearCOp` | `"arith"`, `"div"`, `"qnet"`, `"qcom"`, `"qpa"`, `"qpb"`, `"port"`, `"all"` |
| `NetCOp` | `"set_net"`, `"sync_net"`, `"updt_offset"`, `"set_dt"`, `"get_dt"`, `"set_flag"`, `"get_flag"` |
| `ComCOp` | `"set_flag"`, `"sync"`, `"reset"`, `"set_byte_1/2"`, `"set_hw_1/2"`, `"set_word_1/2"` |
| `ArithCOp` | `"T"`, `"TP"`, `"TM"`, `"PT"`, `"PTP"`, `"PTM"`, `"MT"`, `"MTP"`, `"MTM"` |
| `TrigSrc` | `"set"`, `"clr"` |
| `PACOp` | `"PA"`, `"PB"` |
| `ComFlagVal` | `"0"`, `"1"` |

注意：`WaitInst.c_op` **無預設值**（必填），因為缺少它組譯器會直接 `RuntimeError`。

`CondCode = Literal["Z", "S", "NZ", "NS"]`：`_parse_cond_code()` 保留， 因為有 `ValueError` raise 做早期驗證；QICK assembler 只接受這 4 個大寫值，傳入非法值會在 IR 層立即報錯。

`uf` 欄位型別為 `bool`（不再是 `Optional[str]`）：assembler 只檢查 key  是否存在（`"UF" in command`），值慣例為 `"1"`。`from_dict` 用 `"UF" in d`，`to_dict` 用 `"UF": "1" if self.uf else None`。

## Strict Instruction Field Mapping

- IR 中的 `Instruction` (`BaseInst` 的子類，如 `RegWriteInst`, `PortWriteInst`, `WmemWriteInst` 等) 的 `from_dict` 與 `to_dict` 實作必須**嚴格對應** QICK `tprocv2_assembler.py` 裡的欄位。
- 不可使用任何自定義或延伸的虛擬欄位（例如不該有 `DATA`, `PHASE`, `FREQ` 若底層不支援，將 `DST` 誤植為 `ADDR`）。所有參數（包含 `WW`, `WP` 等 side-effects）都必須精確序列化，以保證最終產生給 QICK 的 `prog_list` 是 合法且無誤的。
