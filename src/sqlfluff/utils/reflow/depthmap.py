"""The DepthMap class is an enriched sequence of raw segments."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from sqlfluff.core.parser import BaseSegment
from sqlfluff.core.parser.segments.base import PathStep
from sqlfluff.core.parser.segments.raw import RawSegment

reflow_logger = logging.getLogger("sqlfluff.rules.reflow")


@dataclass(frozen=True)
class StackPosition:
    """An element of the stack_positions property of DepthInfo."""

    idx: int
    len: int
    type: str

    @staticmethod
    def _stack_pos_interpreter(path_step: PathStep) -> str:
        """Interpret a path step for stack_positions."""
        # If no code, then no.
        if not path_step.code_idxs:
            return ""
        # If there's only one code element, this must be it.
        elif len(path_step.code_idxs) == 1:
            return "solo"
        # Check for whether first or last code element.
        # NOTE: code_idxs is always sorted because of how it's constructed.
        # That means the lowest is always as the start and the highest at the end.
        elif path_step.idx == path_step.code_idxs[0]:
            return "start"
        elif path_step.idx == path_step.code_idxs[-1]:
            return "end"
        else:
            return ""  # NOTE: Empty string evaluates as falsy.

    @classmethod
    def from_path_step(cls, path_step: PathStep) -> "StackPosition":
        """Interpret a PathStep to construct a StackPosition.

        The reason we don't just use the same object is partly
        to interpret it a little more, but also to drop the reference
        to a specific segment which could induce bugs at a later
        stage if used.
        """
        return cls(path_step.idx, path_step.len, cls._stack_pos_interpreter(path_step))


@dataclass(frozen=True)
class DepthInfo:
    """An object to hold the depth information for a specific raw segment."""

    stack_depth: int
    stack_hashes: tuple[int, ...]
    # This is a convenience cache to speed up operations.
    stack_hash_set: frozenset[int]
    stack_class_types: tuple[frozenset[str], ...]
    stack_positions: dict[int, StackPosition]

    @classmethod
    def from_raw_and_stack(
        cls, raw: RawSegment, stack: Sequence[PathStep]
    ) -> "DepthInfo":
        """Construct from a raw and its stack."""
        stack_hashes = tuple(hash(ps.segment) for ps in stack)
        return cls(
            stack_depth=len(stack),
            stack_hashes=stack_hashes,
            stack_hash_set=frozenset(stack_hashes),
            stack_class_types=tuple(ps.segment.class_types for ps in stack),
            stack_positions={
                # Reuse the hash first calculated above.
                stack_hashes[idx]: StackPosition.from_path_step(ps)
                for idx, ps in enumerate(stack)
            },
        )

    def common_with(self, other: "DepthInfo") -> tuple[int, ...]:
        """Get the common depth and hashes with the other."""
        # We use set intersection because it's faster and hashes should be unique.
        common_hashes = self.stack_hash_set.intersection(other.stack_hashes)
        # We should expect there to be _at least_ one common ancestor, because
        # they should share the same file segment. If that's not the case we
        # we should error because it's likely a bug or programming error.
        assert common_hashes, "DepthInfo comparison shares no common ancestor!"
        common_depth = len(common_hashes)
        return self.stack_hashes[:common_depth]

    def trim(self, amount: int) -> "DepthInfo":
        """Return a DepthInfo object with some amount trimmed."""
        # Excluded from coverage: no longer triggered since AL01 rule was refactored
        if amount == 0:  # pragma: no cover
            # The trivial case.
            return self
        new_hash_set = self.stack_hash_set.difference(self.stack_hashes[-amount:])
        return self.__class__(
            stack_depth=self.stack_depth - amount,
            stack_hashes=self.stack_hashes[:-amount],
            stack_hash_set=new_hash_set,
            stack_class_types=self.stack_class_types[:-amount],
            stack_positions={
                k: v for k, v in self.stack_positions.items() if k in new_hash_set
            },
        )


class DepthMap:
    """A mapping of raw segments to depth and parent information.

    This class addresses two needs:
    - To understand configuration of segments with no whitespace
      within them - so the config is related to the parent and
      not the segment)
    - To map the depth of an indent points to apply some precedence
      for where to insert line breaks.

    The internals are structured around a list to do lookups
    and a dict (keyed with the raw segment UUID) to hold the rest.

    """

    def __init__(self, raws_with_stack: Sequence[tuple[RawSegment, list[PathStep]]]):
        self.depth_info = {}
        for raw, stack in raws_with_stack:
            self.depth_info[raw.uuid] = DepthInfo.from_raw_and_stack(raw, stack)

    @classmethod
    def from_parent(cls: type["DepthMap"], parent: BaseSegment) -> "DepthMap":
        """Generate a DepthMap from all the children of a segment.

        Ancestor metadata is built incrementally while walking the tree.
        Immutable path metadata is shared by siblings, while each segment
        receives its own positions dictionary.
        """
        depth_map = cls(raws_with_stack=[])
        if not parent.segments:
            return depth_map

        stack = [(parent, DepthInfo(0, (), frozenset(), (), {}))]
        while stack:
            segment, depth_info = stack.pop()
            if segment.is_type("raw"):
                depth_map.depth_info[segment.uuid] = depth_info
                continue

            # 每个父节点只构造一次祖先序列，子节点共享其中的不可变信息。
            segment_hash = hash(segment)
            stack_hashes = (*depth_info.stack_hashes, segment_hash)
            stack_class_types = (*depth_info.stack_class_types, segment.class_types)
            stack_hash_set = frozenset(stack_hashes)
            child_count = len(segment.segments)
            code_indices = segment._code_indices
            # 逆序入栈，保持原来的源码遍历顺序。
            for idx in range(child_count - 1, -1, -1):
                stack_positions = depth_info.stack_positions.copy()
                stack_positions[segment_hash] = StackPosition.from_path_step(
                    PathStep(segment, idx, child_count, code_indices)
                )
                stack.append(
                    (
                        segment.segments[idx],
                        DepthInfo(
                            stack_depth=depth_info.stack_depth + 1,
                            stack_hashes=stack_hashes,
                            stack_hash_set=stack_hash_set,
                            stack_class_types=stack_class_types,
                            stack_positions=stack_positions,
                        ),
                    )
                )
        return depth_map

    @classmethod
    def from_raws_and_root(
        cls: type["DepthMap"],
        raw_segments: Sequence[RawSegment],
        root_segment: BaseSegment,
    ) -> "DepthMap":
        """Generate a DepthMap a sequence of raws and a root.

        NOTE: This is the less efficient way to construct a DepthMap
        as it doesn't take advantage of caching in the same way as
        `from_parent`.
        """
        buff = []
        for raw in raw_segments:
            stack = root_segment.path_to(raw)
            buff.append((raw, stack))
        return cls(raws_with_stack=buff)

    def get_depth_info(self, raw: RawSegment) -> DepthInfo:
        """Get the depth info for a given segment."""
        try:
            return self.depth_info[raw.uuid]
        except KeyError as err:  # pragma: no cover
            reflow_logger.exception("Available UUIDS: %s", self.depth_info.keys())
            raise KeyError(
                "Tried to get depth info for unknown "
                f"segment {raw} with UUID {raw.uuid}"
            ) from err

    def copy_depth_info(
        self, anchor: RawSegment, new_segment: RawSegment, trim: int = 0
    ) -> None:
        """Copy the depth info for one segment and apply to another.

        This mutates the existing depth map. That's ok because it's
        an idempotent operation and uuids should be unique.

        This is used in edits to a reflow sequence when new segments are
        inserted and can't infer their own depth info.

        NOTE: we don't remove the old one because it causes no harm.
        """
        self.depth_info[new_segment.uuid] = self.get_depth_info(anchor).trim(trim)
