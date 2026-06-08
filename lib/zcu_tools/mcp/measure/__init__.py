"""measure-gui MCP server entry — bridges an MCP host to the live measure-gui's
RemoteControlAdapter over TCP. The wire contract + handlers live with the app under
``zcu_tools.gui.app.main.services.remote``; this package holds only the launchable
``server`` bridge.
"""
