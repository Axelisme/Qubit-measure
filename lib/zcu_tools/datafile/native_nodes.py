"""Local HDF5 lookup without opening external targets."""

import h5py as h5


def known_node(node: h5.Group, name: str) -> h5.Group | h5.Dataset | None:
    """Look up a known relative/absolute path inside node's file.

    Resolve local soft links component by component before dereferencing hard
    links. Missing/dangling paths return None. External links, including soft
    targets through external parents, raise a located ValueError without opening
    the target file. Cyclic/over-nested soft links raise ValueError at HDF5's
    default link traversal limit. Unknown nodes need not call this function.
    """
    location = name if name.startswith("/") else f"{str(node.name).rstrip('/')}/{name}"
    current: h5.Group | h5.Dataset = node.file if name.startswith("/") else node
    pending = name.split("/")
    traversals = 0
    # Match HDF5's own limit rather than imposing a new wire restriction.
    limit = h5.h5p.create(h5.h5p.LINK_ACCESS).get_nlinks()
    while pending:
        part = pending.pop(0)
        if part in {"", "."}:
            continue
        if not isinstance(current, h5.Group):
            return None
        link = current.get(part, getlink=True)
        if link is None:
            return None
        if isinstance(link, h5.ExternalLink):
            raise ValueError(f"{location}: external link targets are not allowed")
        if isinstance(link, h5.SoftLink):
            traversals += 1
            if traversals > limit:
                raise ValueError(f"{location}: cyclic or over-nested soft link")
            pending = link.path.split("/") + pending
            if link.path.startswith("/"):
                current = node.file
            continue
        child = current[part]
        if not isinstance(child, h5.Group | h5.Dataset):
            raise ValueError(f"{location}: expected group or dataset")
        current = child
    return current


def known_group(node: h5.Group, name: str) -> h5.Group:
    """Return/create a local known group; external targets/wrong kinds raise ValueError."""
    child = known_node(node, name)
    if child is None:
        return node.create_group(name, track_order=True)
    if not isinstance(child, h5.Group):
        raise ValueError(f"{str(node.name).rstrip('/')}/{name}: expected group")
    return child
