"""Convex obstacle geometry and independent clearance checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .model import VehicleParams, rotation, translation


@dataclass(frozen=True)
class ConvexPolytope:
    """A compact 2-D convex obstacle ``{p | A p <= b}``."""

    A: np.ndarray
    b: np.ndarray
    label: str = "obstacle"

    def __post_init__(self):
        A = np.asarray(self.A, dtype=float)
        b = np.asarray(self.b, dtype=float)
        if A.ndim != 2 or A.shape[1] != 2 or len(A) < 3:
            raise ValueError("A must have shape (facets, 2) with at least 3 facets")
        if b.shape != (len(A),):
            raise ValueError("b must have one entry per row of A")
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("obstacle coefficients must be finite")
        if np.any(np.linalg.norm(A, axis=1) <= 1e-12):
            raise ValueError("obstacle facet normals must be nonzero")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)


def axis_aligned_rectangle(
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    *,
    label: str = "obstacle",
) -> ConvexPolytope:
    """Create an axis-aligned rectangular obstacle in world coordinates."""
    if width <= 0.0 or height <= 0.0:
        raise ValueError("rectangle width and height must be positive")
    A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    b = np.array(
        [
            center_x + width / 2.0,
            -center_x + width / 2.0,
            center_y + height / 2.0,
            -center_y + height / 2.0,
        ]
    )
    return ConvexPolytope(A, b, label)


def polytope_vertices(obstacle: ConvexPolytope) -> np.ndarray:
    """Return counter-clockwise vertices by intersecting pairs of facets."""
    vertices = []
    for first, second in combinations(range(len(obstacle.A)), 2):
        matrix = obstacle.A[[first, second]]
        if abs(np.linalg.det(matrix)) <= 1e-12:
            continue
        point = np.linalg.solve(matrix, obstacle.b[[first, second]])
        if np.all(obstacle.A @ point <= obstacle.b + 1e-9):
            vertices.append(point)
    if len(vertices) < 3:
        raise ValueError(f"{obstacle.label} is not a bounded nonempty polygon")
    vertices = np.unique(np.round(vertices, decimals=12), axis=0)
    center = np.mean(vertices, axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


def vehicle_vertices(state, params: VehicleParams) -> np.ndarray:
    """Return the four world-frame corners of the rectangular vehicle body."""
    half_length = params.length / 2.0
    half_width = params.width / 2.0
    body = np.array(
        [
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width],
        ]
    )
    return body @ rotation(state).T + translation(state, params)


def _intervals_overlap(first, second) -> bool:
    return max(first[0], second[0]) <= min(first[1], second[1]) + 1e-12


def _polygons_intersect(first: np.ndarray, second: np.ndarray) -> bool:
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.array([-edge[1], edge[0]])
            axis /= np.linalg.norm(axis)
            projection_first = first @ axis
            projection_second = second @ axis
            if not _intervals_overlap(
                (projection_first.min(), projection_first.max()),
                (projection_second.min(), projection_second.max()),
            ):
                return False
    return True


def _point_segment_distance(point, start, end) -> float:
    segment = end - start
    fraction = np.clip(
        float((point - start) @ segment) / float(segment @ segment), 0.0, 1.0
    )
    return float(np.linalg.norm(point - (start + fraction * segment)))


def polygon_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Euclidean distance between two convex polygons; zero means intersection."""
    if _polygons_intersect(first, second):
        return 0.0
    distances = []
    for vertices, edges_of in ((first, second), (second, first)):
        for vertex in vertices:
            for index, start in enumerate(edges_of):
                end = edges_of[(index + 1) % len(edges_of)]
                distances.append(_point_segment_distance(vertex, start, end))
    return min(distances)


def trajectory_clearance(
    states,
    obstacles: tuple[ConvexPolytope, ...],
    params: VehicleParams,
) -> np.ndarray:
    """Independently calculate minimum body-to-obstacle distance at each state."""
    states = np.asarray(states, dtype=float)
    obstacle_vertices = [polytope_vertices(obstacle) for obstacle in obstacles]
    return np.asarray(
        [
            min(
                polygon_distance(vehicle_vertices(state, params), polygon)
                for polygon in obstacle_vertices
            )
            for state in states
        ]
    )
