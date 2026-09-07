import numpy as np

from sitemind.planning import astar


def path_exposure(path, risk):
    return sum(float(risk[cell]) for cell in path)


def test_astar_avoids_hard_obstacle():
    risk = np.zeros((10, 14), dtype=np.float32)
    risk[2:8, 7] = 1.0
    result = astar(risk, (5, 1), (5, 12))
    assert result is not None
    assert all(risk[cell] < 0.99 for cell in result.path)


def test_risk_weight_reduces_exposure():
    risk = np.zeros((11, 15), dtype=np.float32)
    risk[4:7, 4:11] = 0.75
    start, goal = (5, 1), (5, 13)
    shortest = astar(risk, start, goal, risk_weight=0.0)
    safer = astar(risk, start, goal, risk_weight=8.0)
    assert shortest is not None and safer is not None
    assert path_exposure(safer.path, risk) < path_exposure(shortest.path, risk)
