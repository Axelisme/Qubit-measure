"""Session-core UI — app-agnostic dialogs/widgets every measurement-session app
reuses (setup / device / context-inspect / progress).

These depend on the session services through a narrow controller port
(``SessionControllerPort``) and on the session value/event types; they never
reach into a concrete ``gui.app.*`` package. Each app implements the port (its
Controller) and subclasses where it adds app-specific surface (measure adds
ml-edit to the inspect dialog).
"""
