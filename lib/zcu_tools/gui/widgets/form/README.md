# gui/widgets/form — Form draft view

**Last updated:** 2026-10-08，typed discriminator projection

FormEditor renders caller-owned FormState using a host-captured FormEnv. Parsed input writes the draft. Incomplete text and parse errors stay in the Qt view. The generic local_parse_errors query lets hosts reject submission while state still holds an earlier valid value.

The view projects the public input schema and read-only FormPreview. It does not import core walkers or evaluate expressions. Core owns evaluation, model validation, choices and branch defaults. The view owns controls, error display and subscription lifetime. Sequences and typed mappings are whole leaves. Detach and close do not reset or submit the caller's state.

Tagged branch identity comes from the schema owner's x-form-discriminator-values metadata, not the current choices enum. Selector options intersect that identity with current leaf options. Empty choices keep the current branch visible and its input unchanged. Numeric discriminators offer ordinary typed tags without expression mode.

T1TunablesPanel is the concrete workflow caller. Apply, Discard, Start, revision checks and persistence stay with the host. This tool does not migrate CfgSchema or Measure.
