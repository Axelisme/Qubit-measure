# `zcu_tools.gui.measure_cfg` — program cfg GUI vocabulary

**Last updated:** 2026-07-11 — canonical shape catalog

此 Qt-free package 是 program/v2 module/waveform GUI shape 的唯一 owner。`PROGRAM_SHAPES`
固定列出七種 module 與六種 waveform discriminator、label與fresh Spec factory；它不做runtime
registration，也不import program runtime、app、session、experiment、Qt或`meta_tool`。

每次`ProgramShape.make_spec(policy)`都建立deep-fresh tree。`ProgramSpecPolicy`只容許兩個跨app
差異：Arb data的choices source，以及Direct/Pulse Readout間的inheritance hook。main與autoflux
各自在app edge綁定policy；app-specific raw materialization、reference allowed subset與role seed不屬於
catalog。

Catalog lookup對explicit unknown discriminator Fast Fail。raw輸入缺少waveform style時是否採Const
是materializer policy，不是catalog fallback。generic `zcu_tools.gui.cfg`不得反向import本package；
runtime closed set parity由tests顯式比較program cfg classes。
