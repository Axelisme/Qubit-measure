from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BufferKind(str, Enum):
    RUN = "run"
    ANALYZE = "analyze"
    COMMENT = "comment"
    FILE_IMAGE = "file_image"
    FILE_TEXT = "file_text"
    FILE_CSV = "file_csv"


@dataclass
class BufferDescriptor:
    buffer_id: str
    group_id: str
    title: str
    kind: BufferKind
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroupDescriptor:
    group_id: str
    title: str
    buffer_ids: List[str] = field(default_factory=list)
    current_index: int = 0

    def add_buffer(self, buffer_id: str) -> None:
        if buffer_id not in self.buffer_ids:
            self.buffer_ids.append(buffer_id)

    def current_buffer_id(self) -> Optional[str]:
        if not self.buffer_ids:
            return None
        idx = max(0, min(self.current_index, len(self.buffer_ids) - 1))
        self.current_index = idx
        return self.buffer_ids[idx]


@dataclass
class GuiState:
    groups: Dict[str, GroupDescriptor] = field(default_factory=dict)
    buffers: Dict[str, BufferDescriptor] = field(default_factory=dict)
    current_group_id: Optional[str] = None

    def ensure_group(self, group_id: str, title: str) -> GroupDescriptor:
        group = self.groups.get(group_id)
        if group is None:
            group = GroupDescriptor(group_id=group_id, title=title)
            self.groups[group_id] = group
            if self.current_group_id is None:
                self.current_group_id = group_id
        return group

    def add_buffer(self, buffer: BufferDescriptor) -> None:
        self.buffers[buffer.buffer_id] = buffer
        group = self.ensure_group(buffer.group_id, buffer.group_id)
        group.add_buffer(buffer.buffer_id)

    def set_current_group(self, group_id: str) -> None:
        if group_id in self.groups:
            self.current_group_id = group_id

    def current_group(self) -> Optional[GroupDescriptor]:
        if self.current_group_id is None:
            return None
        return self.groups.get(self.current_group_id)

    def current_buffer(self) -> Optional[BufferDescriptor]:
        group = self.current_group()
        if group is None:
            return None
        buf_id = group.current_buffer_id()
        if buf_id is None:
            return None
        return self.buffers.get(buf_id)
