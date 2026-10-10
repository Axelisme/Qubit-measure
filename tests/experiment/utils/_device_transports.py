"""In-memory VISA transport for public device helper contracts."""

from threading import Event


class DeviceTransport:
    """Record SCPI I/O without replacing any driver operations.

    mode selects CURR/VOLT. stop interrupts the first level write when supplied;
    failure injects the exact transport error on the next write.
    """

    def __init__(self, mode: str = "CURR") -> None:
        self.resource_name = "SIMULATED::INSTR"
        self.read_termination = ""
        self.write_termination = ""
        self.mode = mode
        self.output = "1"
        self.level = 0.0
        self.frequency = 1e9
        self.power = -120.0
        self.writes: list[str] = []
        self.stop: Event | None = None
        self.failure: OSError | None = None

    def query(self, command: str) -> str:
        """Return one hardware observation for the declared SCPI commands."""
        values = {
            "*IDN?": "simulated instrument",
            ":SOURce:FUNCtion?": self.mode,
            ":OUTPut?": self.output,
            ":SOURce:LEVel?": str(self.level),
            "SOUR:FREQ?": str(self.frequency),
            "SOUR:POW:POW?": str(self.power),
            ":IQ:STAT?": "0",
        }
        return values[command]

    def write(self, command: str) -> None:
        """Apply a recorded setting, or propagate the supplied transport error."""
        if self.failure is not None:
            raise self.failure
        self.writes.append(command)
        value = command.rsplit(" ", 1)[1]
        if command.startswith(":SOURce:LEVel:AUTO "):
            self.level = float(value)
            if self.stop is not None:
                self.stop.set()
        elif command.startswith(":OUTPut "):
            self.output = value
        elif command.startswith(":SOUR:FREQ "):
            self.frequency = float(value)
        elif command.startswith(":SOUR:POW:POW "):
            self.power = float(value)
        else:
            raise ValueError(f"Unexpected write: {command}")

    def close(self) -> None:
        """The simulated transport owns no external resources."""
