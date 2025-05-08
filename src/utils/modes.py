import math
import networkx as nx
from typing import List, Tuple, Optional

from . import basename, console, get_transformations

SUPPORTED_EXTENSIONS = (".mid", ".midi")


def find_path(
    g: nx.Graph,
    source: str,
    destination: str,
    played_files: List[str],
    max_nodes: int = 5,
    min_nodes: int = 0,
    max_updates: int = 20,
    max_visits: int = 1,
    allow_transpose: bool = True,
    allow_shift: bool = True,
    allow_same_seg: bool = False,
    verbose: bool = False,
) -> Optional[Tuple[List[str], float]]:
    """
    Find the path from source to destination with the smallest average edge cost.

    This function uses a recursive backtracking approach to explore paths from source
    to destination with at most `max_nodes` nodes, where each node can be visited
    up to `max_visits` times. It selects the path with the lowest average edge cost.

    Parameters
    ----------
    G : networkx.Graph
        A weighted graph with a 'weight' attribute on each edge.
    source : str
        The starting node.
    destination : str
        The target node.
    played_files : list[str]
        List of files/nodes that should be avoided in the path.
    max_nodes : int
        The maximum number of nodes allowed in the path. Defaults to 5.
    min_nodes : int
        The minimum number of nodes required in the path. Defaults to 1.
    max_updates : int
        The maximum number of times to update the best path. Defaults to 20.
    max_visits : int
        Maximum number of times a node can be visited. Defaults to 1 for simple paths.
        Set to 0 to run simple djikstra, 2 or higher to allow revisiting nodes.
    allow_same_seg: bool
        If True, allow revisiting transformations of the same segment. Defaults to False.
    verbose: bool
        If True, print verbose output. Defaults to False.

    Returns
    -------
    tuple or None
        A tuple (path, total_cost) where path is a list of nodes representing the
        path with the smallest average edge cost, and total_cost is the sum of
        weights along that path; or None if no such path exists.
    """
    if min_nodes == 0:
        min_nodes = max_nodes - 1
    # handle case where max_visits is 0
    if max_visits < 1:
        # Use weight='weight' for Dijkstra's algorithm
        try:
            path_nodes = nx.shortest_path(
                g, source=source, target=destination, weight="weight"
            )
            # Calculate total cost if needed, otherwise return 0 or actual cost
            # For now, returning 0 as in the original code for this specific case.
            # If the actual cost is needed, it should be calculated, e.g., using nx.shortest_path_length
            path_cost = nx.shortest_path_length(
                g, source=source, target=destination, weight="weight"
            )
            if len(path_nodes) < min_nodes:
                if verbose:
                    console.log(
                        f"dijkstra path length {len(path_nodes)} less than min_nodes {min_nodes}"
                    )
                return None
            return [str(node) for node in path_nodes], path_cost
        except nx.NetworkXNoPath:
            if verbose:
                console.log(
                    f"no path found from '{source}' to '{destination}' using Dijkstra"
                )
            return None

    # convert played_files to set for faster lookups and strip file extensions
    played_nodes = {
        basename(f)
        for f in played_files
        if f != source and basename(f) != basename(destination)
    }
    played_segs = {
        node.split("_")[0] + "_" + node.split("_")[1] for node in played_nodes
    }

    if verbose:
        console.log(f"played_nodes: {played_nodes}")

    # check if destination is in played_files
    if basename(destination) in played_nodes:
        if verbose:
            console.log(f"destination '{destination}' is in played files, skipping")
        return None

    best_path_info = {"path": None, "total_cost": math.inf, "avg_cost": math.inf}
    # counter for number of best path updates
    update_count = {"value": 0}

    def backtrack(current_path, total_weight, visit_counts):
        """
        recursive function to explore paths from source to destination.

        parameters
        ----------
        current_path : list
            the current path being explored.
        total_weight : float
            the total weight of edges in the current path.
        visit_counts : dict
            dictionary tracking number of visits to each node.

        returns
        -------
        bool
            true if we should continue searching, false if we've reached max updates.
        """
        # if we've reached the update limit, stop searching
        if update_count["value"] >= max_updates:
            return False

        current_node = current_path[-1]

        # if we've reached the destination, check if this path has a better average cost
        if current_node == destination:
            # calculate average cost (total weight / number of edges)
            # check if path meets minimum node requirement
            if len(current_path) >= min_nodes:
                num_edges = len(current_path) - 1
                # If num_edges is 0, it means path is just [source] and source == destination
                # In this case, avg_cost should be 0 if min_nodes <= 1, otherwise it's not a valid path yet
                avg_cost = 0 if num_edges == 0 else total_weight / num_edges
                if avg_cost < best_path_info["avg_cost"]:
                    if verbose:
                        console.log(
                            f"		[grey70]found new best path with avg cost {avg_cost:.4f} "
                            f"(update {update_count['value'] + 1}/{max_updates})[/grey70]"
                        )
                    best_path_info["path"] = current_path.copy()
                    best_path_info["total_cost"] = total_weight
                    best_path_info["avg_cost"] = avg_cost
                    update_count["value"] += 1
                    if update_count["value"] >= max_updates:
                        if verbose:
                            console.log(
                                f"		[grey70]reached maximum of {max_updates} path updates, "
                                f"returning best path found[/grey70]"
                            )
                        return False
            return True

        # if we've exceeded max_nodes, stop exploring this path
        if len(current_path) >= max_nodes:
            return True

        # explore neighbors
        # Use g.neighbors(current_node) or g[current_node].items() based on preference/need for edge data
        for neighbor in g.neighbors(current_node):
            data = g.get_edge_data(current_node, neighbor)
            weight = data.get("weight", 1)  # Default weight if not specified

            # Check transformations
            neighbor_transformations = get_transformations(neighbor)
            if (
                not allow_transpose
                and neighbor_transformations[1].get("transpose", 0) != 0
            ):
                continue
            if not allow_shift and neighbor_transformations[1].get("shift", 0) != 0:
                continue

            # Check if segment was played
            neighbor_seg = neighbor.split("_")[0] + "_" + neighbor.split("_")[1]
            if not allow_same_seg and neighbor_seg in played_segs:
                # if verbose:
                #     console.log(
                #         f"		[grey70]skipping transformation of played segment {neighbor}[/grey70]"
                #     )
                continue

            # skip if in played_nodes or if we've visited this node max times
            neighbor_basename = basename(neighbor)
            if (
                neighbor_basename not in played_nodes
                and visit_counts.get(neighbor, 0) < max_visits
            ):

                # pruning: if adding this edge would already make the average cost worse than the best found,
                # don't explore this path further (only if we have found at least one path to destination)
                if best_path_info["path"] is not None:
                    # current edges + 1 for this new edge
                    potential_edges = len(current_path)
                    potential_avg = (total_weight + weight) / potential_edges
                    if potential_avg >= best_path_info["avg_cost"]:
                        continue

                # increment visit count for this neighbor
                visit_counts[neighbor] = visit_counts.get(neighbor, 0) + 1

                # add neighbor to path and continue exploration
                current_path.append(neighbor)
                should_continue = backtrack(
                    current_path, total_weight + weight, visit_counts
                )
                current_path.pop()  # backtrack

                # decrement visit count when backtracking
                visit_counts[neighbor] -= 1
                if visit_counts[neighbor] == 0:
                    del visit_counts[neighbor]

                # stop exploring if we've reached max updates
                if not should_continue:
                    return False

        return True

    # start backtracking from source with initial visit count for source node
    initial_visits = {source: 1}
    backtrack([source], 0, initial_visits)

    if best_path_info["path"] is None:
        if verbose:
            console.log(
                f"no path found from '{source}' to '{destination}' with {min_nodes} <= nodes <= {max_nodes} "
                f"and max {max_visits} visits per node"
            )
        return None

    if verbose:
        console.log(
            f"found path with {len(best_path_info['path'])} nodes, total cost {best_path_info['total_cost']:.4f}, "
            f"and average edge cost {best_path_info['avg_cost']:.4f} after {update_count['value']} updates"
        )

    return best_path_info["path"], best_path_info["total_cost"]
