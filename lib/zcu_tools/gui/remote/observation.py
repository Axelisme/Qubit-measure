"""Declarative owner-thread resource observation policy for remote methods."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceObservationPolicy:
    """Declare a method's per-connection resource-version contract.

    ``guard_deps`` lists required resource key templates, expanded from validated
    request params. A trailing ``*`` matches both current and previously seen keys
    with that prefix. Missing observations are stale, including at version zero.
    ``reveals`` lists complete resources read by a successful reply, expanded from
    original request params. ``reveals_without`` names params whose explicit
    presence makes a read partial; ``reveals_when_nonempty`` names params that
    must have truthy original values before a read establishes observations.
    ``refresh_after_write`` advances only previously seen, matching resources
    changed by a successful handler. ``created_resource`` is an existence key
    template expanded from its reply; ``created_identity`` names the reply's
    required nonempty string identity. Creation certifies only a 0-to-1 change.
    Empty tuples and False declare no observation behavior. Both creation fields
    default to None and must be supplied together with write tracking enabled.
    Invalid conditional-read or creation declarations raise ValueError.
    """

    guard_deps: tuple[str, ...] = ()
    reveals: tuple[str, ...] = ()
    reveals_without: tuple[str, ...] = ()
    reveals_when_nonempty: tuple[str, ...] = ()
    refresh_after_write: bool = False
    created_resource: str | None = None
    created_identity: str | None = None

    def __post_init__(self) -> None:
        if (self.reveals_without or self.reveals_when_nonempty) and not self.reveals:
            raise ValueError("conditional reveals require revealed resources")
        if (self.created_resource is None) != (self.created_identity is None):
            raise ValueError("creation requires both a resource and an identity field")
        if self.created_resource is not None and (
            not self.created_resource
            or not self.created_identity
            or not self.refresh_after_write
        ):
            raise ValueError("created_resource requires identity and write tracking")
