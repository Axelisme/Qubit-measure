# gui/widgets/form — Form draft view

**Last updated:** 2026-10-11，string expression controls

FormEditor renders caller-owned FormState using a host-captured FormEnv. Parsed input writes the draft. Incomplete text and parse errors stay in the Qt view. The generic local_parse_errors query lets hosts reject submission while state still holds an earlier valid value.

The view projects the public input schema and read-only FormPreview. It does not import core walkers or evaluate expressions. Core owns evaluation, model validation, choices and branch defaults. The view owns controls, error display and subscription lifetime. Sequences and typed mappings are whole leaves. Detach and close do not reset or submit the caller's state. discard_local_text drops view-local unparseable text and redraws the current state. The host separately resets its draft on Discard; the renderer does not own that policy.

Tagged branch identity comes from the schema owner's x-form-discriminator-values metadata, not the current choices enum. Selector options intersect that identity with current leaf options. Empty choices keep the current branch visible and its input unchanged. All discriminators are structural selectors: they determine model shape and branch defaults. Their schema and controls offer only ordinary typed tags, without expression mode.

String value leaves, including Literal[str] and string Enum values, offer Direct and Expression modes like numeric fields. Direct text stays literal, including text starting with $. Expression mode writes an explicit envelope to FormState. The view displays the core-resolved preview as `$flux_dev = "yoko1"` and shows alias-naming errors without fallback. It never replaces the authored expression with the resolved name.

T1TunablesPanel is the concrete workflow caller. Apply, Discard, Start, revision checks and persistence stay with the host. This tool does not migrate CfgSchema or Measure.
