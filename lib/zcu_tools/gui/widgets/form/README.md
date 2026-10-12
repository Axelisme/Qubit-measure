# gui/widgets/form — Form draft view

**Last updated:** 2026-10-13，no Direct echo, one-dropdown choices, dense aligned rows

FormEditor renders caller-owned FormState using a host-captured FormEnv. Parsed input writes the draft. Incomplete text and parse errors stay in the Qt view. The generic local_parse_errors query lets hosts reject submission while state still holds an earlier valid value.

The view projects the public input schema and read-only FormPreview. It does not import core walkers or evaluate expressions. Core owns evaluation, model validation, choices and branch defaults. The view owns controls, error display and subscription lifetime. Sequences and typed mappings are whole leaves. Detach and close do not reset or submit the caller's state. discard_local_text drops view-local unparseable text and redraws the current state. The host separately resets its draft on Discard; the renderer does not own that policy.

Tagged branch identity comes from the schema owner's x-form-discriminator-values metadata, not the current choices enum. Selector options intersect that identity with current leaf options. Empty choices keep the current branch visible and its input unchanged. All discriminators are structural selectors: they determine model shape and branch defaults. Their schema and controls offer only ordinary typed tags, without expression mode.

String value leaves, including Literal[str] and string Enum values, offer Direct and Expression modes like numeric fields. Direct text stays literal, including text starting with $. Expression mode writes an explicit envelope to FormState. The view displays the core-resolved preview as `$flux_dev = "yoko1"` and shows alias-naming errors without fallback. It never replaces the authored expression with the resolved name.

The value label shows only an Expression-mode resolved value; Direct input is not echoed. Empty value and error labels are hidden and take no space; field errors appear under their row. A choice leaf with expression support (Literal, string Enum or a choices entry) shows one dropdown in Direct mode: the current choices, then an `Expression…` entry. Its mode combo appears only in Expression mode. Returning to Direct selects the resolved value when it is a current choice, otherwise the last Direct choice. Discriminators keep their plain tag dropdown.

Rows sit 4 px apart and nested groups use small margins. The labels of one model level share a left-aligned column as wide as its widest label, so that level's controls start at one left edge; each nested group has its own column. The editor has no outer margins; hosts add them. One stretch below the last row takes spare height, so nested groups do not grow.

T1TunablesPanel is the concrete workflow caller. Apply, Discard, Start, revision checks and persistence stay with the host. This tool does not migrate CfgSchema or Measure.
