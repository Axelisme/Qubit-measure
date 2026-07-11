# `zcu_tools.gui.measure_cfg` — program cfg GUI vocabulary

**Last updated:** 2026-07-11 — strict shape inspection

此 Qt-free package 是 program/v2 module/waveform GUI shape 的唯一 owner。`PROGRAM_SHAPES`
固定列出七種 module 與六種 waveform discriminator、label與fresh Spec factory；它不做runtime
registration，也不import program runtime、app、session、experiment、Qt或`meta_tool`。

每次`ProgramShape.make_spec(policy)`都建立deep-fresh tree。`ProgramSpecPolicy`只容許兩個跨app
差異：Arb data的choices source，以及Direct/Pulse Readout間的inheritance hook。main與autoflux
各自在app edge綁定policy；runtime object normalization與role seed仍不屬於本package。

`ProgramMaterializationPolicy`把generic `gui.cfg` spec walk綁成program raw contract：missing
`ch`/`ro_ch`為0、其它scalar為unset；missing或non-mapping nested section建立完整Spec default；required
reference missing選`allowed[0]`。main可materialize完整7+6 shapes；autoflux module subset只含Pulse與
Pulse Readout，但label lookup仍辨識完整legal catalog。Bath module-local `relax_delay`明確拒絕，
program root同名欄位不在此materializer scope。

Catalog lookup對explicit unknown discriminator Fast Fail。raw輸入缺少waveform style時是否採Const
由materializer policy固定為Const，不是catalog fallback。generic `zcu_tools.gui.cfg`不得反向import本package；
runtime closed set parity由tests顯式比較program cfg classes。

`program_shape_for_input(kind, cfg_input)`是root-only inspection seam：Mapping只查root key，typed
object只讀`type`/`style` attribute，不呼`to_dict()`、Spec factory或materializer。missing、non-string與
unknown discriminator都Fast Fail；Library enumeration使用inspection，只有resolve/edit才走各app
materializer façade。
