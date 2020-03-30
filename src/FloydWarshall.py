from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def floyd_warshall(graph: np.ndarray, path_reconstruction=False) -> Union[np.ndarray,
                                                                          Optional[Tuple[np.ndarray, np.ndarray]]]:
    """Finds all shortest paths in a weighted graph. Works with positive and negative weights,
    but not with negative cycles.

    Time complexity O(|V|^3)
    Space complexity O(|V|^3)

    Args:
        graph (np.ndarray): Adjacency matrix of the graph as numpy ndarray.
        path_reconstruction (bool, optional): Whether to calculate and output the matrix needed for path reconstruction.

    Return:
        np.matrix: Adjacency matrix showing all shortest paths.
        np.matrix (optional): Adjacency matrix needed for path reconstruction.

    Examples:
        >>> graph = np.asarray([[0, np.inf, -2, np.inf], [4, 0, 3, np.inf], [np.inf, np.inf, 0, 2], [np.inf, -1, np.inf, 0]])
        >>> shortest_paths = floyd_warshall(graph)
        >>> print(shortest_paths)
        [[ 0. -1. -2.  0.]
         [ 4.  0.  2.  4.]
         [ 5.  1.  0.  2.]
         [ 3. -1.  1.  0.]]
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


def shortest_path(graph: np.ndarray, *waypoints, pretty=False) -> Union[str, None, Tuple[List[int], Dict[Tuple[int, int], int]]]:
    """Reconstructs shortest path between multiple nodes

    Args:
        graph (np.matrix): Adjacency matrix of the graph as numpy matrix.
        waypoints: waypoints between which the shortest path should get calculated.
        pretty (bool): Whether to pretty print path or return raw path

    Note:
        if not path exists between all of the waypoints not path will get calculated

    Return:
        str: Formatted text output showing input, shortest path and cost of the shortest path
        Tuple[List[int], Dict[Tuple[int, int], int]]: Tuple list with indices of shortest path and dictionary which
        contains the cost for every step of the shortest path

    Example:
        >>> graph = np.asarray([[0, np.inf, -2, np.inf], [4, 0, 3, np.inf], [np.inf, np.inf, 0, 2], [np.inf, -1, np.inf, 0]])
        >>> path = shortest_path(graph, 0, 1, 3)
        >>> print(path)
        ([0, 2, 3, 1, 0, 2, 3], {(0, 2): -2.0, (2, 3): 2.0, (3, 1): -1.0, (1, 0): 4.0})
    """
    dist_matrix, path_matrix = floyd_warshall(graph, path_reconstruction=True)

    assert len(waypoints) > 0, "No waypoints supplied."
    assert all(isinstance(waypoint, int) for waypoint in waypoints), "False datatype for at least one waypoint " \
                                                                     "(int needed)"

    for i in range(len(waypoints) - 1):
        try:
            path_matrix[waypoints[i]][waypoints[i + 1]]
        except IndexError:
            return "IndexError: Certain waypoint nodes are not part of the input graph"

    def calculate_shortest_path():
        path = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            u = waypoints[i]
            v = waypoints[i + 1]
            while u != v:
                u = path_matrix[u][v]
                path.append(u)

        return path

    def pprint_path():
        _waypoints = " ⟶ ".join([str(node) for node in waypoints])
        _shortest_path = " ⟶ ".join([str(node) for node in path])
        _cost = '\n'.join(
            [f"{node[0]} ⟶ {node[1]} ({dist_matrix[node[0]][node[1]]})" for node in list(zip(path, path[1:]))])
        return f"== Waypoints ==\n{_waypoints}\n\n==Shortest Path ==\n{_shortest_path}\n\n== Cost ==\n{_cost}"

    def calc_path_costs():
        return {(node[0], node[1]): dist_matrix[node[0]][node[1]] for node in list(zip(path, path[1:]))}

    path = calculate_shortest_path()
    if pretty:
        return print(f"{pprint_path()}\nTotal Cost: {sum(calc_path_costs().values())}")
    return path, calc_path_costs()
