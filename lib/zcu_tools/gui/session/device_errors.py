"""Qt-free device failures shared by the device service and context facet."""


class DeviceRegistrationError(RuntimeError):
    """Driver construction or registration failed or a named device is unavailable."""
