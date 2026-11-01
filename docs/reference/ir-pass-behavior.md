# IR pass behavior

**Readers:** framework-dev, agent. Reference for IR pass conditions, compiler output and optimization guards.

Pipeline ownership and ordering remain in the [IR README](../../lib/zcu_tools/program/v2/ir/README.md#pipeline).
Test construction and acceptance rules remain in [tests](../../tests/README.md#dispatch-與最佳化-guard-驗收).

- `UnrollLoopPass` builds a jump table for a register-driven loop only when
  `available_regs` is nonempty, `body_size > 0` and the budget/timing factor
  `k_final >= 2`. It rounds that factor down to a power of two. The timing factor
  and `pmem_budget // body_size` jointly bound the expansion.
- `SimplifyDispatchPass` lowers a two-target `IRDispatch` to a `BlockNode` with
  conditional and unconditional basic blocks. A nonzero selector jumps to
  `target_labels[1]`; zero jumps to `target_labels[0]`. Both jumps are explicit.
  Small PMEM uses direct labels, and `BranchEliminationPass` may remove the
  unconditional jump when its target immediately follows. Big PMEM uses
  `REG_WR s15 label` plus `JUMP [s15]` for both targets and retains both jumps.
- `DmemDispatchPass` uses `needs_big_jump(pmem_capacity)` for its out-of-range
  guard. At capacity <= 2048, the guard is a label-mode `JumpInst` with
  `label=LabelRef(last)` and `addr=None`. Above 2048, it writes the last label to
  s15 and uses `JumpInst(addr=s15)`. The guard tests `value_reg - n` with `NS`;
  an index >= n selects the last target. The lookup then reads the selected
  program address from DMEM and jumps indirectly.
- `DeadWriteEliminationPass` and `ZeroDelayDCEPass` preserve a block marked
  `disable_opt=True` instead of applying their local optimization.
