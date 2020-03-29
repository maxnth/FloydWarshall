from typing import List, Optional, Union

import numpy as np


def floyd_warshall(graph: np.ndarray, path_reconstruction=False) -> Union[np.ndarray, Optional[(np.ndarray, np.ndarray)]]:
    """Finds all shortest paths in a weighted graph. Works with positive and negative weights,
    but not with negative cycles.

    Time complexity O(|V|^3)
    Space complexity O(|V|^3)

    Args:
        graph (np.ndarray): Adjacency matrix of the graph as numpy ndarray.
        path_reconstruction (bool): Whether to calculate and output the matrix needed for path reconstruction.
    Return:
        np.matrix: Adjacency matrix showing all shortest paths.
        np.matrix (optional): Adjacency matrix needed for path reconstruction.

    """
    assert graph.shape[0] == graph.shape[1], "Input matrix must be a square adjacency matrix"

    dist_matrix = np.full(graph.shape, np.inf)
    if path_reconstruction:
        next_matrix = np.full(graph.shape, 0)

    for edge, weight in np.ndenumerate(graph):
        dist_matrix[edge[0], edge[1]] = weight

        if path_reconstruction:
            next_matrix[edge[0], edge[1]] = edge[1]

    np.fill_diagonal(dist_matrix, 0)
    if path_reconstruction:
        np.fill_diagonal(next_matrix, range(next_matrix.shape[0]))

    for k in range(0, graph.shape[0]):
        for i in range(0, graph.shape[0]):
            for j in range(0, graph.shape[0]):
                if dist_matrix[i][j] > dist_matrix[i][k] + dist_matrix[k][j]:
                    dist_matrix[i][j] = dist_matrix[i][k] + dist_matrix[k][j]
                    if path_reconstruction:
                        next_matrix[i][j] = next_matrix[i][k]

    if path_reconstruction:
        return dist_matrix, next_matrix
    return dist_matrix


def shortest_path(graph: np.ndarray, u: int, v: int) -> List[int]:
    """Reconstructs shortest path between multiple nodes

    Args:
        graph (np.matrix): Adjacency matrix of the graph as numpy matrix.
        u (int): Starting node for the path.
        v (int): Target node of the path.
    """
    dist_matrix, path_matrix = floyd_warshall(graph, path_reconstruction=True)

    try:
        path_matrix[u][v]
    except IndexError:
        return []

    path = [u]

    while u != v:
        u = path_matrix[u][v]
        print(u)
        path.append(u)
    return path