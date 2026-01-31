"""
Coordinate transformation utilities for personalized genomes.

Maps reference genome coordinates to personalized genome coordinates,
accounting for insertions/deletions caused by genetic variants.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import bisect


@dataclass
class CoordinateTransformer:
    """
    Maps reference genome coordinates to personalized genome coordinates.

    Uses a sparse shift map: only stores positions where shifts change (at variant sites).
    Binary search lookup: O(log n) where n = number of variants.

    Example:
        >>> # After deletion of 5bp at position 100
        >>> t = CoordinateTransformer([(0, 0), (100, -5)])
        >>> t.ref_to_personal(150)  # Returns 145
    """
    shift_points: List[Tuple[int, int]] = field(default_factory=list)
    # Each tuple: (ref_position, cumulative_shift_after_this_position)

    def __post_init__(self):
        """Sort shift points and build lookup arrays."""
        if self.shift_points:
            self.shift_points.sort(key=lambda x: x[0])
        self._positions = [p for p, _ in self.shift_points]
        self._shifts = [s for _, s in self.shift_points]

    def ref_to_personal(self, ref_pos: int) -> int:
        """
        Transform reference position to personalized genome position.

        Args:
            ref_pos: Position in reference genome coordinates

        Returns:
            Position in personalized genome coordinates
        """
        if not self._positions:
            return ref_pos  # No variants = identity transform

        # Binary search: find rightmost position <= ref_pos
        idx = bisect.bisect_right(self._positions, ref_pos) - 1

        if idx < 0:
            return ref_pos  # Before first variant

        return ref_pos + self._shifts[idx]

    def transform_interval(self, ref_start: int, ref_end: int) -> Tuple[int, int]:
        """
        Transform both boundaries of an interval independently.

        Args:
            ref_start: Interval start in reference coordinates
            ref_end: Interval end in reference coordinates

        Returns:
            (pers_start, pers_end) in personalized coordinates
        """
        return self.ref_to_personal(ref_start), self.ref_to_personal(ref_end)

    @property
    def total_shift(self) -> int:
        """Net shift at end of region (for length calculation)."""
        return self._shifts[-1] if self._shifts else 0
