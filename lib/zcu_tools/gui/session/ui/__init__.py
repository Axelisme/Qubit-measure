"""Session-core UI — app-agnostic dialogs/widgets every measurement-session app
reuses (setup / device / context-inspect / progress).

Setup/predictor/inspect depend on the session services through a narrow
controller port (``SessionControllerPort``); device management depends on the
device-control facet plus a separate md provider for eval-mode fields. They
never reach into a concrete ``gui.app.*`` package. Each app implements/exposes
the needed ports and subclasses where it adds app-specific surface (measure adds
ml-edit to the inspect dialog).
"""
