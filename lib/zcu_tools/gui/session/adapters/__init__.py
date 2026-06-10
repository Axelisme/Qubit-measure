"""Session-core driven adapters — concrete Qt/OS implementations of the session
ports that every measurement-session app reuses.

Unlike the session leaf modules these DO pull in Qt (a driven adapter is the Qt
side of a port), so import them only when building the UI, after the matplotlib
backend is selected.
"""
