from __future__ import annotations

import numpy as np

from copula_causal_sim.graphs.dag import validate_dag


def sample_random_dag(
    num_nodes: int,
    edge_prob: float,
    max_indegree: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample a random DAG with bounded indegree.

    Method:
    1. sample a random topological order,
    2. only allow edges from earlier to later nodes in that order,
    3. sample candidate edges with probability edge_prob,
    4. enforce max_indegree per child.

    Convention:
        adj[i, j] = 1 means i -> j
    """
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")

    if not (0.0 <= edge_prob <= 1.0):
        raise ValueError("edge_prob must be between 0 and 1.")

    if max_indegree < 0:
        raise ValueError("max_indegree must be non-negative.")

    rng = np.random.default_rng(seed)

    order = rng.permutation(num_nodes)
    adj = np.zeros((num_nodes, num_nodes), dtype=int)

    for child_position in range(1, num_nodes):
        child = int(order[child_position])
        possible_parents = order[:child_position].astype(int).tolist()

        rng.shuffle(possible_parents)

        selected_parents: list[int] = []

        for parent in possible_parents:
            if len(selected_parents) >= max_indegree:
                break

            if rng.random() < edge_prob:
                selected_parents.append(parent)

        for parent in selected_parents:
            adj[parent, child] = 1

    validate_dag(adj, max_allowed_indegree=max_indegree)

    return adj


def sample_random_dags(
    num_graphs: int,
    num_nodes: int,
    edge_prob: float,
    max_indegree: int,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Sample multiple random DAGs.

    Uses a master seed and creates independent seeds for each graph.
    """
    if num_graphs <= 0:
        raise ValueError("num_graphs must be positive.")

    rng = np.random.default_rng(seed)
    graphs: list[np.ndarray] = []

    for _ in range(num_graphs):
        graph_seed = int(rng.integers(0, 2**32 - 1))

        adj = sample_random_dag(
            num_nodes=num_nodes,
            edge_prob=edge_prob,
            max_indegree=max_indegree,
            seed=graph_seed,
        )

        graphs.append(adj)

    return graphs