import numpy as np
import pytest

from copula_causal_sim.graphs.dag import (
    adjacency_to_edge_list,
    is_dag,
    max_indegree,
    parents_of,
    summarize_dag,
    topological_order,
    validate_dag,
)
from copula_causal_sim.graphs.random_graphs import sample_random_dag


def test_simple_dag_validation():
    adj = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    validated = validate_dag(adj, max_allowed_indegree=2)

    assert validated.shape == (3, 3)
    assert is_dag(validated)
    assert max_indegree(validated) == 2


def test_cycle_detection():
    adj = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    with pytest.raises(ValueError):
        validate_dag(adj)


def test_topological_order_respects_edges():
    adj = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    order = topological_order(adj)
    position = {node: idx for idx, node in enumerate(order)}

    for parent, child in adjacency_to_edge_list(adj):
        assert position[parent] < position[child]


def test_parents_of():
    adj = np.array([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ])

    assert parents_of(adj, 0) == []
    assert parents_of(adj, 1) == [0]
    assert parents_of(adj, 2) == [0, 1]


def test_random_dag_respects_max_indegree():
    adj = sample_random_dag(
        num_nodes=20,
        edge_prob=0.4,
        max_indegree=3,
        seed=123,
    )

    validate_dag(adj, max_allowed_indegree=3)

    assert adj.shape == (20, 20)
    assert max_indegree(adj) <= 3


def test_summarize_dag():
    adj = np.array([
        [0, 1],
        [0, 0],
    ])

    summary = summarize_dag(adj)

    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert summary["max_indegree"] == 1
    assert summary["edges"] == [(0, 1)]