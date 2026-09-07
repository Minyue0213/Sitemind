"""Grid-based risk-aware A* used by the first SiteMind MVP."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, inf
from typing import Dict, List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]


@dataclass(frozen=True)
class AStarResult:
    path: List[Cell]
    cost: float
    expanded_nodes: int


def _validate_cell(name: str, cell: Cell, shape: Tuple[int, int]) -> None:
    row, col = cell
    if not (0 <= row < shape[0] and 0 <= col < shape[1]):
        raise ValueError(f"{name} {cell} lies outside grid {shape}")


def astar(
    risk_map: np.ndarray,
    start: Cell,
    goal: Cell,
    *,
    risk_weight: float = 5.0,
    obstacle_threshold: float = 0.99,
    allow_diagonal: bool = True,
) -> Optional[AStarResult]:
    """Plan a path whose edge cost trades distance against predicted risk."""

    risk = np.asarray(risk_map, dtype=np.float32)
    if risk.ndim != 2:
        raise ValueError("risk_map must be a 2D array")
    risk = np.nan_to_num(risk, nan=1.0, posinf=1.0, neginf=1.0)
    if risk.min() < 0 or risk.max() > 1:
        raise ValueError("risk_map values must lie in [0, 1]")
    if risk_weight < 0:
        raise ValueError("risk_weight cannot be negative")
    _validate_cell("start", start, risk.shape)
    _validate_cell("goal", goal, risk.shape)
    blocked = risk >= obstacle_threshold
    if blocked[start] or blocked[goal]:
        return None

    cardinal = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    diagonal = [(-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)]
    moves = cardinal + diagonal if allow_diagonal else cardinal

    frontier: list[Tuple[float, int, Cell]] = []
    sequence = 0
    heappush(frontier, (hypot(goal[0] - start[0], goal[1] - start[1]), sequence, start))
    came_from: Dict[Cell, Cell] = {}
    cost_so_far: Dict[Cell, float] = {start: 0.0}
    expanded = 0

    while frontier:
        _, _, current = heappop(frontier)
        expanded += 1
        if current == goal:
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return AStarResult(path, cost_so_far[goal], expanded)

        for d_row, d_col, distance in moves:
            neighbor = (current[0] + d_row, current[1] + d_col)
            row, col = neighbor
            if not (0 <= row < risk.shape[0] and 0 <= col < risk.shape[1]):
                continue
            if blocked[neighbor]:
                continue
            if d_row and d_col:
                if blocked[current[0] + d_row, current[1]] or blocked[current[0], current[1] + d_col]:
                    continue
            mean_risk = 0.5 * (float(risk[current]) + float(risk[neighbor]))
            new_cost = cost_so_far[current] + distance * (1.0 + risk_weight * mean_risk)
            if new_cost >= cost_so_far.get(neighbor, inf):
                continue
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = current
            heuristic = hypot(goal[0] - row, goal[1] - col)
            sequence += 1
            heappush(frontier, (new_cost + heuristic, sequence, neighbor))
    return None
