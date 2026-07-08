from collections import defaultdict
import math
from typing import Dict, List


class UnionFind:
    """
    Union-Find (disjoint set) for transitive merge propagation.
    Ensures that if A→B and B→C are both valid merges, all three
    collapse to a single canonical ID rather than being applied
    as two independent pairs.
    """

    def __init__(self):
        self.parent: Dict[int, int] = {}
        self.rank: Dict[int, int] = defaultdict(int)

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self) -> Dict[int, List[int]]:
        """Return {root_id: [all member ids]} for every group with >1 member."""
        result: Dict[int, List[int]] = defaultdict(list)
        for x in self.parent:
            result[self.find(x)].append(x)
        return {root: members for root, members in result.items() if len(members) > 1}
    
    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0


class MotionSnapshot:
    """Lightweight summary of a track's physical motion profile."""

    __slots__ = (
        "player_id",
        "frame_start",
        "frame_end",
        "timestamp_start",
        "timestamp_end",
        "detection_count",
        "mean_dx",
        "mean_dy",
        "mean_speed_kmh",
        "mean_acceleration",
        "first_x1",
        "first_y1",
        "first_x2",
        "first_y2",
        "last_x1",
        "last_y1",
        "last_x2",
        "last_y2",
    )

    def __init__(
        self,
        player_id: int,
        frame_start: int,
        frame_end: int,
        timestamp_start: float,
        timestamp_end: float,
        detection_count: int,
        mean_dx: float,
        mean_dy: float,
        mean_speed_kmh: float,
        mean_acceleration: float,
        first_x1: float,
        first_y1: float,
        first_x2: float,
        first_y2: float,
        last_x1: float,
        last_y1: float,
        last_x2: float,
        last_y2: float,
    ) -> None:
        self.player_id = player_id
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.detection_count = detection_count
        self.mean_dx = mean_dx
        self.mean_dy = mean_dy
        self.mean_speed_kmh = mean_speed_kmh
        self.mean_acceleration = mean_acceleration
        self.first_x1 = first_x1
        self.first_y1 = first_y1
        self.first_x2 = first_x2
        self.first_y2 = first_y2
        self.last_x1 = last_x1
        self.last_y1 = last_y1
        self.last_x2 = last_x2
        self.last_y2 = last_y2

    @property
    def direction_angle_deg(self) -> float:
        """Predominant heading of the track in degrees [0, 360)."""
        return math.degrees(math.atan2(self.mean_dy, self.mean_dx)) % 360
