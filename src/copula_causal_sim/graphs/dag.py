from __future__ import annotations

from collections import deque

import numpy as np


def as_adjacency_matrix(adjacency: np.ndarray) -> np.ndarray:
    """
    Convert input to a clean integer adjacency matrix.

    Convention:
        adjacency[i, j] = 1 means i -> j
    """
    adj = np.asarray(adjacency)

    if adj.ndim != 2:
        raise ValueError(f"Adjacency matrix must be 2D, got shape {adj.shape}.")

    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {adj.shape}.")

    if not np.all((adj == 0) | (adj == 1)):
        raise ValueError("Adjacency matrix must contain only 0 and 1 values.")

    adj = adj.astype(int)

    if np.any(np.diag(adj) != 0):
        raise ValueError("Adjacency matrix must not contain self-loops.")

    return adj


def num_nodes(adjacency: np.ndarray) -> int:
    adj = as_adjacency_matrix(adjacency)
    return adj.shape[0]


def parents_of(adjacency: np.ndarray, node: int) -> list[int]:
    adj = as_adjacency_matrix(adjacency)

    if node < 0 or node >= adj.shape[0]:
        raise IndexError(f"Node index out of range: {node}")

    return np.where(adj[:, node] == 1)[0].tolist()


def children_of(adjacency: np.ndarray, node: int) -> list[int]:
    adj = as_adjacency_matrix(adjacency)

    if node < 0 or node >= adj.shape[0]:
        raise IndexError(f"Node index out of range: {node}")

    return np.where(adj[node, :] == 1)[0].tolist()


def indegrees(adjacency: np.ndarray) -> np.ndarray:
    adj = as_adjacency_matrix(adjacency)
    return adj.sum(axis=0)


def outdegrees(adjacency: np.ndarray) -> np.ndarray:
    adj = as_adjacency_matrix(adjacency)
    return adj.sum(axis=1)


def max_indegree(adjacency: np.ndarray) -> int:
    return int(indegrees(adjacency).max())


def topological_order(adjacency: np.ndarray) -> list[int]:
    """
    Return a topological order using Kahn's algorithm.

    Raises ValueError if the graph contains a directed cycle.
    """
    adj = as_adjacency_matrix(adjacency)
    n = adj.shape[0]

    indeg = adj.sum(axis=0).astype(int)
    queue = deque(np.where(indeg == 0)[0].tolist())

    order: list[int] = []

    while queue:
        node = queue.popleft()
        order.append(node)

        for child in np.where(adj[node, :] == 1)[0]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(int(child))

    if len(order) != n:
        raise ValueError("Graph is not a DAG. A directed cycle was detected.")

    return order


def is_dag(adjacency: np.ndarray) -> bool:
    try:
        topological_order(adjacency)
        return True
    except ValueError:
        return False


def validate_dag(
    adjacency: np.ndarray,
    max_allowed_indegree: int | None = None,
) -> np.ndarray:
    """
    Validate an adjacency matrix as a DAG.

    Optionally checks a maximum allowed indegree.
    """
    adj = as_adjacency_matrix(adjacency)

    # This raises if cyclic.
    topological_order(adj)

    if max_allowed_indegree is not None:
        observed = max_indegree(adj)

        if observed > max_allowed_indegree:
            raise ValueError(
                f"Graph max indegree is {observed}, "
                f"but max_allowed_indegree={max_allowed_indegree}."
            )

    return adj


def adjacency_to_edge_list(adjacency: np.ndarray) -> list[tuple[int, int]]:
    adj = as_adjacency_matrix(adjacency)

    edges: list[tuple[int, int]] = []

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] == 1:
                edges.append((i, j))

    return edges


def summarize_dag(adjacency: np.ndarray) -> dict:
    adj = validate_dag(adjacency)

    return {
        "num_nodes": int(adj.shape[0]),
        "num_edges": int(adj.sum()),
        "max_indegree": max_indegree(adj),
        "max_outdegree": int(outdegrees(adj).max()),
        "topological_order": topological_order(adj),
        "edges": adjacency_to_edge_list(adj),
    }