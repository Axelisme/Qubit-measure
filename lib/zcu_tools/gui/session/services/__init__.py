"""Session-core services — the measurement-session application services shared by
the GUI apps (connection / device / context / startup).

App-agnostic: these depend only on the session value types/events/state and on
narrow ports (``gui/session/ports``) that each app's composition root fills with
its own concrete infrastructure (the exclusion gate, the background executor, the
progress hub, project IO, the driver factory). ``build_session_services`` wires
the session bundle; an app's ``build_app_services`` calls it and adds its own
experiment-surface services.
"""
