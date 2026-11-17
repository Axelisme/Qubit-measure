"""Session-core UI — app-agnostic dialogs/widgets every measurement-session app
reuses (entry / device / context-inspect / progress).

Entry depends on the session services through ``EntryControlPort``;
context-inspect, device, and predictor management depend on their own control
facets. Device management also receives a separate md provider for eval-mode
fields. They never reach into a concrete ``gui.app.*`` package. Each app
exposes the needed ports and subclasses where it adds app-specific surface
(measure adds ml-edit to the inspect dialog).
"""
